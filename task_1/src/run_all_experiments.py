"""Показать или выполнить полный граф повторных экспериментов."""
import argparse
import json
import os

from notebook_workflow import NotebookWorkflow


EXPERIMENTS = ('eda', 'baseline', 'bootstrap', 'rolling', 'optuna', 'selection',
               'used', 'full', 'controls', 'direct-intervals', 'test-plan',
               'diagnostics', 'corrected', 'channel-control', 'feature-analysis', 'formula-benchmark')


def ordered_stages(workflow, requested):
    result = []
    def visit(stage):
        for dependency in workflow.dependencies(stage):
            visit(dependency)
        if stage not in result:
            result.append(stage)
    for stage in requested:
        visit(stage)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='repeat_01')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--experiments', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--raw-events')
    parser.add_argument('--artist-mapping')
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args(argv)
    workflow = NotebookWorkflow('recompute', args.run_name, args.download,
                                args.raw_events, args.artist_mapping, args.include_test)
    requested = list(args.experiments)
    if args.include_test:
        if set(requested).intersection(['full', 'controls', 'test-plan', 'diagnostics']):
            requested.append('old-test')
        if set(requested).intersection(['corrected', 'feature-analysis']):
            requested.append('corrected-test')
    stages = ordered_stages(workflow, requested)
    if not args.execute:
        records = []
        for stage in stages:
            try:
                commands = workflow.commands(stage)
            except (ValueError, FileNotFoundError) as error:
                commands = [str(error)]
            records.append({'stage': stage, 'commands': commands})
        print(json.dumps({'executed': False, 'run_name': args.run_name,
                          'test_requested': args.include_test, 'stages': records},
                         ensure_ascii=False, indent=2))
        return
    for stage in stages:
        workflow.ensure(stage)
    print(json.dumps({'completed': True, 'run_name': args.run_name, 'stages': stages}, ensure_ascii=False))


if __name__ == '__main__':
    main()
