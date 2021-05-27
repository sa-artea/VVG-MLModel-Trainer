"""
* Copyright 2020, Maestria de Humanidades Digitales,
* Universidad de Los Andes
*
* Developed for the Msc graduation project in Digital Humanities
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# =========================================
# Standard library imports
# =========================================
import os
import sys
import configparser

# workaround of the relative explicit import limitations
# altering the sys.path Keep checking the '../..' parameter
# to go further into the app buildpath
file_path = os.path.join(os.path.dirname(__file__), '..')
file_dir = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.abspath(file_path))
data_dir = os.path.join(file_dir, 'Data')


def load_config(*args):
    """
    Read an INI file to load the configuration into data structure

    Returns:
        ans (configparser): loaded configuration to execute the app
    """

    cfg_fp = os.path.join(*args)
    ans = configparser.ConfigParser()
    ans.read(cfg_fp, encoding="utf-8")
    return ans

# config = configparser.ConfigParser()
# with open('example.ini', 'w', encoding="utf-8") as configfile:
#     config.write(configfile)
