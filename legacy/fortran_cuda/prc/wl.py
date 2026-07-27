#!/usr/bin/env python
import argparse
import os
import sys


PATH = {
    'job': os.path.join('job'),
    'src': os.path.join('src'),
    'prc': os.path.join('prc'),
}


def _load_app(app_name):
    aliases = {
        'debug': 'run',
        'make': 'build',
    }
    app_name = aliases.get(app_name, app_name)

    if app_name == 'create':
        from app_create import create as app
    elif app_name == 'remove':
        from app_remove import remove as app
    elif app_name == 'list':
        from app_list import list as app
    elif app_name == 'edit':
        from app_edit import edit as app
    elif app_name == 'clean':
        from app_clean import clean as app
    elif app_name == 'pre':
        from app_pre import pre as app
    elif app_name == 'build':
        from app_build import build as app
    elif app_name == 'solve':
        from app_solve import solve as app
    elif app_name == 'run':
        from app_run import run as app
    elif app_name == 'monitor1':
        from app_monitor1 import monitor1 as app
    elif app_name == 'monitor2':
        from app_monitor2 import monitor2 as app
    elif app_name == 'post':
        from app_post import post as app
    elif app_name == 'post0':
        from app_post0 import post as app
    elif app_name == 'post1':
        from app_post1 import post as app
    elif app_name == 'post2':
        from app_post2 import post as app
    elif app_name == 'anime':
        from app_anime import anime as app
    else:
        print(' ERROR: no app named ' + app_name)
        sys.exit(1)

    return app


def print_start():
    print('########################################################')
    print('# WiRE-LES')
    print('########################################################')
    print('\n')


def print_info(app_name, case_name):
    print('# app  = ' + app_name)
    print('# case = ' + case_name)
    print('\n')


def print_end():
    print('\n')
    print('########################################################')


def main():
    parser = argparse.ArgumentParser(description='wireles')
    parser.add_argument('app')
    parser.add_argument('case')
    args = parser.parse_args()

    print_start()
    print_info(args.app, args.case)
    app = _load_app(args.app)
    app(PATH, args.case)
    print_end()


if __name__ == '__main__':
    main()
