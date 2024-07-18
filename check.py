# -*- coding: utf-8 -*-
# @Author  : xuelun

import csv
from os import listdir
from os.path import join

home = join('dump', 'zeb')

# specified_key2 = "GL3D"
specified_keys = [
    'GL3D', 'KITTI', 'ETH3DI', 'ETH3DO', 'GTASfM', 'ICLNUIM', 'MultiFoV',
    'SceneNet', 'BlendedMVS', 'RobotcarNight', 'RobotcarSeason', 'RobotcarWeather'
]

for specified_key2 in specified_keys:
    identifiers_dict = {}

    for filename in listdir(home):
        if filename.endswith(".txt") and ']' in filename:
            parts = filename[:-4].split()
            if parts[2] == specified_key2:
                with open(join(home, filename), 'r') as f:
                    reader = csv.reader(f, delimiter=' ')
                    file_identifiers = [row[0] for row in reader if row]
                    identifiers_dict[filename] = file_identifiers

    all_identical = True
    reference_identifiers = None
    if identifiers_dict:
        reference_identifiers = list(identifiers_dict.values())[0]
        for identifiers in identifiers_dict.values():
            if identifiers != reference_identifiers:
                all_identical = False
                break

    if all_identical:
        print("Good ! all {} file identifiers is same".format(specified_key2))
    else:
        print("Bad ! file {} have different identifiers".format(specified_key2))

    if not all_identical:
        for filename, identifiers in identifiers_dict.items():
            if identifiers != reference_identifiers:
                print(f"File {filename} 's {specified_key2} identifiers is different with others")
