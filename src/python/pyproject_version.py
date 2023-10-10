# NOTE: the current version of scikit-build-core is incomplete and 
# however, it is the recommended way for customizing version strings 
# Maybe we will submit a PR to fix this issue

from pathlib import Path
import os
from setuptools_scm.version import guess_next_version, ScmVersion


def myversion_func(version: ScmVersion):
    #res2 = version.format_next_version(guess_next_version, '{guessed}b{distance}')  #default
    version_file = Path(__file__).parent / 'VERSION'
    assert version_file.exists()
    base_version = version_file.read_text().strip()
    git_hash = version.node  
    # git_branch = version.branch  # not used, but available
    stamp = version.node_date.strftime("%y%m%d")   # the recent git commit date
    res = base_version + '+'  + stamp  + '+' + git_hash 
    return res
