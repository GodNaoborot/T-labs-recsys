"""Повторить ручные YetiRank и MultiClass с исходными 42 признаками."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time

import catboost
from catboost import CatBoostRanker, CatBoostClassifier, Pool
from catboost.utils import eval_metric
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import log_loss

PROJECT = Path(__file__).resolve().parents[1]
SEED, THREADS = 42, 4
NDCG = 'NDCG:top=10;type=Base;denominator=LogPosition'
METADATA = {'uid', 'item_id', 'start_ts', 'split', 'label', 'group_id'}


def run_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise argparse.ArgumentTypeError('Имя: 1–64 буквы, цифры, _ и -')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_split(path, names, split):
    """Сортировать целые группы и исключить группы с постоянной меткой."""
    frame = pq.read_table(path, columns=names+['label','group_id'],
                          filters=[('split','==',split)]).sort_by('group_id').to_pandas()
    if frame.empty:
        raise ValueError(f'Пустая выборка: {split}')
    labels, groups = frame.pop('label'), frame.pop('group_id')
    starts = np.flatnonzero(np.r_[True, groups.to_numpy()[1:] != groups.to_numpy()[:-1]])
    sizes = np.diff(np.r_[starts,len(groups)])
    mixed = np.minimum.reduceat(labels.to_numpy(),starts) != np.maximum.reduceat(labels.to_numpy(),starts)
    keep = np.repeat(mixed,sizes)
    stats = {'rows_total':len(labels),'groups_total':len(starts),
             'rows_used':int(keep.sum()),'groups_used':int(mixed.sum())}
    if not keep.any():
        raise ValueError(f'Нет групп с различными метками: {split}')
    return (frame.loc[keep].reset_index(drop=True),
            labels.loc[keep].reset_index(drop=True),
            groups.loc[keep].reset_index(drop=True),stats)


def simple_scores(frame, seed=SEED):
    """Повторить случайный, константный и семь статистических baseline."""
    ties = np.random.default_rng(seed).permutation(len(frame))
    scores = {'random ranking':ties.astype('float64'),'class prior':np.zeros(len(frame))}
    columns = {'item popularity 7d':'item_lst_amt_7',
               'item popularity 30d':'item_lst_amt_30',
               'user listened track 30d':'ui_lst_amt_30',
               'user track decay 7d':'ui_lst_decay_7',
               'user liked track':'ui_liked_now',
               'user listened artist 30d':'ua_lst_amt_30'}
    for name,column in columns.items():
        scores[name] = frame[column].fillna(0).to_numpy(copy=True)
    scores['user listened track recently'] = -frame['ui_time_since_last_lst'].fillna(1e9).to_numpy(copy=True)
    return scores,ties


def quantize(train, val, output):
    """Обучить пороги на train и применить к validation."""
    started = time.perf_counter()
    train.quantize(border_count=254,random_seed=SEED,nan_mode='Min',used_ram_limit='6GB')
    train.save_quantization_borders(str(output/'train_borders.tsv'))
    val.quantize(input_borders=str(output/'train_borders.tsv'),nan_mode='Min',used_ram_limit='6GB')
    return time.perf_counter()-started


def ranking_metrics(labels, groups, score, ties):
    order = np.lexsort((ties,np.asarray(score,dtype='float64')))
    ranks = np.empty(len(score),dtype='float64')
    ranks[order] = np.arange(len(score))
    return {'NDCG@10':eval_metric(labels,ranks,NDCG,group_id=groups,thread_count=THREADS)[0],
            'QueryAUC':eval_metric(labels,score,'QueryAUC:type=Ranking',group_id=groups,thread_count=THREADS)[0]}


def fit_one(objective, train, val, groups, model_path, names):
    """Обучить, сохранить и сверить прогнозы загруженного checkpoint."""
    common = dict(random_seed=SEED,learning_rate=.1,iterations=600,
                  early_stopping_rounds=50,use_best_model=True,verbose=25,
                  allow_writing_files=False)
    ranker = objective == 'YetiRank'
    estimator = (CatBoostRanker(loss_function='YetiRank',eval_metric=NDCG,
                                thread_count=4,**common) if ranker else
                 CatBoostClassifier(loss_function='MultiClass',eval_metric='MultiClass',
                                    thread_count=8,**common))
    started = time.perf_counter()
    estimator.fit(train,eval_set=val)
    elapsed = time.perf_counter()-started
    started = time.perf_counter()
    estimator.save_model(str(model_path))
    save_seconds = time.perf_counter()-started
    if estimator.feature_names_ != names:
        raise ValueError('Порядок признаков модели отличается от входа')
    end = int(np.searchsorted(groups.to_numpy(),groups.iloc[min(2047,len(groups)-1)],side='right'))
    probe = val.slice(np.arange(end))
    loaded = type(estimator)().load_model(str(model_path))
    method = 'predict' if ranker else 'predict_proba'
    np.testing.assert_array_equal(getattr(estimator,method)(probe),getattr(loaded,method)(probe))
    record = {'mode':'trained_with_validation','path':str(model_path),
              'sha256':sha256(model_path),'params':estimator.get_all_params(),
              'fit_seconds':elapsed,'load_seconds':None,'save_seconds':save_seconds,
              'training_threads':estimator.get_params()['thread_count'],
              'tree_count':estimator.tree_count_,'best_iteration':estimator.get_best_iteration()}
    return estimator,record


def train(args):
    """Выполнить подготовку двух Pool, два fit и оценку на validation."""
    path = args.features.resolve()
    names = [column for column in pq.read_schema(path).names if column not in METADATA]
    if len(names) != 42:
        raise ValueError('Нужны 42 ручных входа')
    models = PROJECT/'models/baselines/runs'/args.run_name
    output = PROJECT/'results/baselines/runs'/args.run_name
    if models.exists() or output.exists():
        raise FileExistsError('Имя уже использовано; нужен новый --run-name')
    models.mkdir(parents=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    x_train,y_train,g_train,train_stats = load_split(path,names,'train')
    x_val,y_val,g_val,val_stats = load_split(path,names,'val')
    classes = np.array([0,1,2])
    np.testing.assert_array_equal(np.sort(y_train.unique()),classes)
    prior = y_train.value_counts(normalize=True).reindex(classes).to_numpy()
    scores,ties = simple_scores(x_val)
    train_pool = Pool(x_train,label=y_train,group_id=g_train)
    val_pool = Pool(x_val,label=y_val,group_id=g_val)
    del x_train,x_val
    preparation = {'load_and_pool_seconds':time.perf_counter()-started}
    preparation['quantization_seconds'] = quantize(train_pool,val_pool,output)
    estimators,records = {},{}
    for objective,filename in [('YetiRank','ranker.cbm'),('MultiClass','pointwise.cbm')]:
        estimators[objective],records[objective] = fit_one(
            objective,train_pool,val_pool,g_val,models/filename,names)
    ranker,pointwise = estimators['YetiRank'],estimators['MultiClass']
    scores['catboost YetiRank'] = ranker.predict(val_pool,thread_count=THREADS)
    proba = pointwise.predict_proba(val_pool,thread_count=THREADS)
    scores['catboost MultiClass, E[label]'] = proba @ pointwise.classes_
    metrics = pd.DataFrame({name:ranking_metrics(y_val,g_val,score,ties)
                            for name,score in scores.items()}).T
    metrics.loc['class prior','logloss'] = -np.log(prior)[y_val.to_numpy().astype(int)].mean()
    metrics.loc['catboost MultiClass, E[label]','logloss'] = log_loss(y_val,proba,labels=classes)
    metrics.index.name = 'model'
    metrics.sort_values('NDCG@10',ascending=False).to_csv(output/'validation_metrics.csv')
    importance = ranker.get_feature_importance(type='PredictionValuesChange',prettified=True)
    validation_importance = ranker.get_feature_importance(val_pool,type='LossFunctionChange',
                                                         prettified=True,thread_count=THREADS)
    importance.merge(validation_importance,on='Feature Id',
                     suffixes=('_share_%','_val_loss_change')).to_csv(output/'feature_importance.csv',index=False)
    config = {'random_seed':SEED,'features':names,'split_stats':{'train':train_stats,'val':val_stats},
              'drop_constant_groups':True,'evaluation_split':'val','test_used':False,
              'training_threads_by_model':{'YetiRank':4,'MultiClass':8},
              'preparation_times':preparation,'timing_scope':'fit only, prequantized Pool',
              'ndcg':NDCG,'ndcg_ties':'fixed-seed random permutation, exact score ties only',
              'query_auc':'QueryAUC:type=Ranking; tied scores count as 0.5',
              'native_ranker_ndcg':eval_metric(y_val,scores['catboost YetiRank'],NDCG,
                                              group_id=g_val,thread_count=THREADS)[0],
              'class_prior':dict(zip(map(str,classes),prior.tolist())),
              'evaluated_dataset':{'path':str(path),'sha256':sha256(path)},'models':records,
              'versions':{'numpy':np.__version__,'pandas':pd.__version__,
                          'pyarrow':pyarrow.__version__,'catboost':catboost.__version__,
                          'sklearn':sklearn.__version__}}
    write_json(output/'run_config.json',config)
    timing = pd.DataFrame.from_dict(records,orient='index')[
        ['mode','fit_seconds','load_seconds','save_seconds','training_threads','tree_count']]
    timing.index.name = 'model'
    timing.to_csv(output/'model_times.csv')
    for objective,record in records.items():
        write_json(output/f'{objective.lower()}_training.json',record)
    print(metrics.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True,type=run_name)
    parser.add_argument('--features',type=Path,default=PROJECT/'data/features.parquet')
    train(parser.parse_args(argv))


if __name__ == '__main__':
    main()
