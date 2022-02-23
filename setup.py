#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup
from setuptools.config import read_configuration

setup_cfg = Path(__file__).parent.joinpath("setup.cfg")
conf_dict = read_configuration(setup_cfg)

url = conf_dict["metadata"]["url"]
version = conf_dict["metadata"]["version"]

setup(download_url=f"{url}/archive/refs/tags/{version}.tar.gz")
