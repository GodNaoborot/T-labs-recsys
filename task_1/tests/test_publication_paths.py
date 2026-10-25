"""Публикуемые отчёты и архивы не должны содержать домашние пути автора."""
from pathlib import Path
import re
import subprocess
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HOME_PATH = re.compile(rb"/home/[A-Za-z0-9_.-]+/|[A-Za-z]:[/\\]+Users[/\\]+", re.IGNORECASE)


class PublicationPathTests(unittest.TestCase):
    def test_published_reports_and_archives_have_no_personal_paths(self):
        files = subprocess.check_output(
            ['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT)
        failures = []
        for item in sorted(set(files.split(b'\0'))):
            if not item:
                continue
            path = ROOT / item.decode()
            if path.suffix == '.zip':
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if HOME_PATH.search(archive.read(member)):
                            failures.append(f'{path.relative_to(ROOT)}:{member}')
            elif path.suffix in {'.json', '.csv', '.md', '.ipynb', '.txt', '.tsv', '.sha256'}:
                if HOME_PATH.search(path.read_bytes()):
                    failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main()
