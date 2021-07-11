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
# =======================================================
# native python libraries
# =======================================================
import sys
import re
import os
import copy

# =======================================================
# extension python libraries
# =======================================================
# NONE IN VIEW

# =======================================================
# developed python libraries
# =======================================================
import conf
from App.controller import Controller
from App.model import Trainer
assert Controller
assert Trainer
assert conf
assert re

# =======================================================
#  Global data config
# =======================================================

CFG_FOLDER = "Config"
CFG_APP = "train-config.ini"
CFG_SCHEMA = "df-schema.ini"
CFG_WEB_TAGS = "html-tags.ini"
CFG_DATA_APP = conf.configGlobal(CFG_FOLDER, CFG_APP)

# url query for VVG gallery request
# vvg_search = CFG_DATA_APP.get("Requests", "small")
# vvg_search = CFG_DATA_APP.get("Requests", "large")
vvg_search = CFG_DATA_APP.get("Requests", "extensive")

print("=================== Config INI Gallery Search ===================")
print(str(vvg_search) + "\n\n")

# root URL of the gallery
vvg_url = CFG_DATA_APP.get("Requests", "root")

# local root dirpath where the data CHANGE IN .INI!!!!
vvg_localpath = CFG_DATA_APP.get("Paths", "localPath")

# real rootpath to work with
galleryf = os.path.normpath(vvg_localpath)
print("================== Config INI Local Path Gallery ==================")
print(galleryf)
print(os.path.isdir(vvg_localpath))

# subdirs in the local root path needed to process data
paintf = CFG_DATA_APP.get("Paths", "paintsFolder")
lettersf = CFG_DATA_APP.get("Paths", "lettersFolder")

# scrapped subfolder
srcf = CFG_DATA_APP.get("Paths", "sourceFolder")

# app subfoder
dataf = CFG_DATA_APP.get("Paths", "dataFolder")
imgf = CFG_DATA_APP.get("Paths", "imageFolder")

# cresting the export file for the data
bfn = CFG_DATA_APP.get("ExportFiles", "basicfile")
fext = CFG_DATA_APP.get("ExportFiles", "fext")

# change according to the request in the .INI!!!!
# fsize = CFG_DATA_APP.get("ExportFiles", "small")
# fsize = CFG_DATA_APP.get("ExportFiles", "large")
fsize = CFG_DATA_APP.get("ExportFiles", "extensive")

expf = bfn + fsize + "." + fext
print("================== Config INI Export File Name ==================")
print(str(expf) + "\n\n")

# loading config schema into the program
dfschema = conf.configGlobal(CFG_FOLDER, CFG_SCHEMA)

# setting schema for the element/paint gallery dataframe
VVG_DF_COLS = eval(dfschema.get("DEFAULT", "columns"))

# column names for creating a new index and model in the program
WC = VVG_DF_COLS[VVG_DF_COLS.index(
    "ID"):VVG_DF_COLS.index("COLLECTION_URL")+1]
index_start_cols = copy.deepcopy(WC)

print("================== Columns for a new DF-Schema ==================")
print(index_start_cols, "\n")

# column names for creating the JSON in the folders
json_index_cols = copy.deepcopy(VVG_DF_COLS[VVG_DF_COLS.index(
    "DESCRIPTION"):VVG_DF_COLS.index("RELATED_WORKS")+1])

print("================= JSON Columns in the DF-Schema =================")
print(json_index_cols, "\n")

# =======================================================
#  data input to start creating index and scraping
# =======================================================

# dummy vars for the index of the dataframe
id_col = str(VVG_DF_COLS[VVG_DF_COLS.index("ID")])
title_col = str(VVG_DF_COLS[VVG_DF_COLS.index("TITLE")])
curl_col = str(VVG_DF_COLS[VVG_DF_COLS.index("COLLECTION_URL")])
dl_col = str(VVG_DF_COLS[VVG_DF_COLS.index("DOWNLOAD_URL")])
haspic_col = str(VVG_DF_COLS[VVG_DF_COLS.index("HAS_PICTURE")])
desc_col = str(VVG_DF_COLS[VVG_DF_COLS.index("DESCRIPTION")])
search_col = str(VVG_DF_COLS[VVG_DF_COLS.index("SEARCH_TAGS")])
obj_col = str(VVG_DF_COLS[VVG_DF_COLS.index("OBJ_DATA")])
rwork_col = str(VVG_DF_COLS[VVG_DF_COLS.index("RELATED_WORKS")])
img_col = str(VVG_DF_COLS[VVG_DF_COLS.index("IMG_DATA")])
shape_col = str(VVG_DF_COLS[VVG_DF_COLS.index("IMG_SHAPE")])


class View():
    """
    the View is the console interface for the program, connect to the
    Model() with the Controller()

    The view is in charge of the interaction with the user selecting
    the options to scrap the web and load the data into the CSV files.
    """
    # ====================================================
    # class variables
    # ====================================================
    gallery_controller = Controller()
    gallery_model = Trainer()
    localg_path = str()
    imgd_path = str()
    webg_path = str()
    schema = VVG_DF_COLS

    # config file for scraped html tags
    scrapy_cfg = None

    # input variables
    inputs = -1

    def __init__(self, *args):
        """
        Class View creator

        Args:
            webg_path (str): URL of the museum gallery
            localg_path (str): the local directory to save the data
            scrapy_cfg (str): costume INI file with HTML tags to scrap

        Raises:
            exp: raise a generic exception if something goes wrong

        """
        try:
            # generic creation
            self.gallery_model = Trainer()
            self.gallery_controller = Controller()
            self.localg_path = str()
            self.imgd_path = str()
            self.webg_path = str()
            self.schema = copy.deepcopy(VVG_DF_COLS)
            self.scrapy_cfg = conf.configGlobal(CFG_FOLDER, CFG_WEB_TAGS)
            self.inputs = -1

            # if args parameters are input in the creator
            if len(args) > 0:

                for i in range(0, len(args)-1):
                    if i == 0:
                        self.webg_path = args[i]

                    if i == 1:
                        self.localg_path = args[i]

                    if i == 2:
                        self.imgd_path = args[i]

                    if i == 3:
                        self.scrapy_cfg = conf.configGlobal(args[i])

                wg = self.webg_path
                gp = self.localg_path
                ip = self.imgd_path
                mod = self.gallery_model
                self.gallery_model = Trainer(wg, gp, ip)
                sch = self.schema
                self.gallery_controller = Controller(wg, gp, ip,
                                                     model=mod,
                                                     schema=sch)

        # exception handling
        except Exception as exp:
            raise exp

    def menu(self):
        """
        menu options console display

        Raises:
            exp: raise a generic exception if something goes wrong
        """

        try:
            print("\n====================== WELCOME ======================\n")
            # create a new index based in the root url
            print("1) Start gallery index (scrap ID, TITLE & COLLECTION_URL)")
            # save in files all the scrapped data
            print("2) Save gallery data (saving into *.CSV)")
            # load preavious scraped data into model
            print("3) Load gallery data (loading from *.CSV)")
            # load preavious scraped data into model
            print("4) Check gallery data (reading current *.CSV)")
            # recovers the basic data from the gallery query
            print("5) Get Gallery elements description (DESCRIPTION)")

            # complement the basic data created from option 6) and 12)
            print("6) Get Gallery elements download URLs (DOWNLOAD_URL)")
            print("7) Download Gallery elements image files (HAS_PICTURE)")
            print("8) Get Gallery elements search-tags (SEARCH_TAGS)")
            print("9) Get Gallery elements collection-data (OBJ_DATA)")
            print("10) Get Gallery elements related work (RELATED_WORKS)")
            print("11) Process Gallery images (IMG_DATA, IMG_SHAPE)")
            print("12) Export DataFrame to JSON Files (from CSV to Local dir)")
            print("99) Auto script for options (3, 5, 6, 7, 8, 9, 10, 11, 12)")
            print("0) EXIT (last option)")
            # finish program

        # exception handling
        except Exception as exp:
            raise exp

    def setup(self):
        """
        Configuring the view class in the main

        Raises:
            exp: print the exception and restart the setup() method
        """
        try:
            # setting gallery base webpage
            self.webg_path = vvg_search
            gf = galleryf
            sf = srcf
            pf = paintf
            cf = os.getcwd()
            df = dataf
            igf = imgf

            # setting up local dir for saving data
            self.localg_path = self.gallery_controller.setup_local(gf, sf, pf)
            self.imgd_path = self.gallery_controller.setup_local(cf, df, igf)
            print("============== Creating the Gallery View ==============")
            print("View gallery localpath: " + str(self.localg_path))
            print("View images localpath: " + str(self.imgd_path))
            print("View gallery Web URL: " + str(self.webg_path))
            print("\n")

            # creating the gallery model
            wg = self.webg_path
            gp = self.localg_path
            ip = self.imgd_path
            vdfc = VVG_DF_COLS

            self.gallery_model = Trainer(wg, gp, ip, schema=vdfc)
            print("============== Creating Gallery Model ==============")
            print("Model gallery localpath: " +
                  str(self.gallery_model.localg_path))
            print("Model images localpath: " +
                  str(self.gallery_model.imgd_path))
            print("Model gallery Web URL: " +
                  str(self.gallery_model.webg_path))
            print("\n")

            gm = self.gallery_model

            # creating the gallery controller
            self.gallery_controller = Controller(wg, gp, ip,
                                                 model=gm,
                                                 schema=vdfc)
            print("============ Crating Gallery Controller ============")
            print("Controller gallery localpath: " +
                  str(self.gallery_controller.localg_path))
            print("Controller images localpath: " +
                  str(self.gallery_controller.imgd_path))
            print("Controller Web gallery URL: " +
                  str(self.gallery_controller.webg_path))
            print("\n")

        # exception handling
        except Exception as exp:
            print(exp)
            self.setup()

    def get_wtags(self, column):
        """
        gets the HTML tags from a config file needed by beatifulsoup to
        create the dataframe column with the same name

        Args:
            column (str): name of the column to get the HTML tags

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (list): list with the 4 HTML tags in the following order:
                - divs: main HTML tag to look for in the scrap
                - attrs: optional HTML tags to look for with the main tag
                - elements: main HTML tag to recover in the scrap
                - cleanup: optional HTML tags for clean up scraped data
        """
        try:
            # default ans for the method
            ans = (None, None, None, None)
            cfg = self.scrapy_cfg

            # checking config file
            if cfg.has_section(column):

                # prepping all to process config file
                ans = list()
                # get all keys in an option
                keys = cfg.options(column)
                # get datatype from first key
                types = cfg.get(column, keys[0])
                # eval() the type list and removing the first key
                types = eval(types)
                keys.pop(0)

                # iterating the column keys and types
                for k, t in zip(keys, types):
                    # getting column, option value
                    temp = cfg.get(column, k)

                    # ifs for different types
                    if t in (dict, list, tuple, None):
                        temp = eval(temp)
                    elif t is int:
                        temp = int(temp)
                    elif t is float:
                        temp = float(temp)
                    elif t is str:
                        temp = str(temp)
                    ans.append(temp)

            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def get_itags(self, column):
        """
        gets the image tags from a config file needed to process the files
        to Black & White (B&W) and in color (RGB)

        Args:
            column (str): name of the column to get the image tags

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (list): list of 1 or 2 image tags in the following order:
                - fext: file extension to save the file
                - rgb: size of the shape np.array for color images
                - bw: size of the shape np.array for b&w images
        """
        try:
            # default ans for the method
            ans = (None, None, None)
            cfg = self.scrapy_cfg

            # checking config file
            if cfg.has_section(column):

                # prepping all to process config file
                ans = list()
                # get all keys in an option
                keys = cfg.options(column)
                # get datatype from first key
                types = cfg.get(column, keys[0])
                # eval() the type list and removing the first key
                types = eval(types)
                keys.pop(0)

                # iterating the column keys and types
                for k, t in zip(keys, types):
                    # getting column, option value
                    temp = cfg.get(column, k)

                    # ifs for different types
                    if t in (dict, list, tuple, None):
                        temp = eval(temp)
                    elif t is str:
                        temp = str(temp)
                    ans.append(temp)

            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def one(self, *args):
        """
        Option 1, it creates a new dataframe with new IDs, Tittles and
        gallery URLs to further scrap data from

        Args:
            id_col (str): df-schema ID column name
            title_col (str): df-schema TITLE column name
            curl_col (str): df-schema COLLECTION column name
            vvg_url (str): gallery URL search for the collection

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Starting a new Gallery (ID, TITLE, COLLECTION_URL)")
            print("...")

            gc = self.gallery_controller
            wg = self.webg_path
            gp = self.localg_path

            # starting the gallery index (gain) from scratch
            id_in = self.get_wtags(args[0])
            gain = gc.scrapidx(wg, 5.0, id_in[0], id_in[1])
            id_data = gc.get_idxid(gain, id_in[2], id_in[3])
            print("Gallery IDs were processed...")

            ti_in = self.get_wtags(args[1])
            gain = gc.scrapagn(ti_in[0], ti_in[1])
            title_data = gc.get_idxtitle(gain, ti_in[2])
            print("Gallery Titles were processed...")

            url_in = self.get_wtags(args[2])
            gain = gc.scrapagn(url_in[0], url_in[1])
            url_data = gc.get_idxurl(gain, args[3], url_in[2])
            print("Gallery collection URLs were processed...")

            data = (id_data, title_data, url_data)
            ans = gc.newdf(args, data)
            print("New Gallery Model was created...")
            gc.create_localfolders(gp, args[0])
            print("Local Gallery folders were created...")
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def two(self, *args):
        """
        Option 2, saves the in-memory data into CSV and creates the
        local dirpath for the files if it doesnt exists

        Args:
            expf (str): export file name, Default CSV
            dataf (str): data folder name for the app

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Saving gallery Model into CSV file...")
            gc = self.gallery_controller
            ans = gc.save_gallery(args[0], args[1])
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def three(self, *args):
        """
        Option 3, loads the in memory of the CSV data and creates the
        local dirpath for the files if it doesnt exists

        Args:
            id_col (str): df-schema column name of the ID
            expf (str): export file name, Default CSV
            dataf (str): data folder name for the app

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Loading Gallery CSV file into Model...")

            gc = self.gallery_controller
            gp = self.localg_path
            ans = gc.load_gallery(args[0], args[1])
            gc.create_localfolders(gp, args[2])
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def four(self):
        """
        Option 4, checks the in memory dataframe

        Raises:
            exp: raise a generic exception if something goes wrong
        """
        try:
            print("Checking Gallery Model status (dataframe from CSV)")

            gc = self.gallery_controller
            gc.check_gallery()

        # exception handling
        except Exception as exp:
            raise exp

    def five(self, *args):
        """
        Option 5, based on the results of option 1, it scrap the
        description of each URL gallery element

        Args:
            desc_col (str): df-schema column name of the DESCRIPTION
            curl_col (str): df-schema column name of the COLLECTION

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Recovering elements description (DESCRIPTION)")

            gc = self.gallery_controller
            opt_in = self.get_wtags(args[0])
            descrip_data = gc.scrap_descriptions(
                args[1],
                opt_in[0],
                opt_in[1],
                opt_in[2],
                opt_in[3],
                multiple=True)

            ans = gc.updata(args[0], descrip_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def six(self, *args):
        """
        Option 6, based on the results of option 1, it scrap the
        image download URL each gallery element

        Args:
            dl_col (str): df-schema column name of the DOWNLOAD_URL
            curl_col (str): df-schema column name of the COLLECTION
            vvg_url (str): web gallery URL search for the collection

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Recovering pictures download urls (DOWNLOAD_URL)")

            gc = self.gallery_controller
            opt_in = self.get_wtags(args[0])
            urlpic_data = gc.scrap_paintlinks(
                args[1],
                args[2],
                opt_in[0],
                opt_in[1],
                opt_in[2],
                multiple=False)

            ans = gc.updata(args[0], urlpic_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def seven(self, *args):
        """
        Option 7, based on the results of option 1, it download the
        actual image from each gallery element

        Args:
            dl_col (str): df-schema column name of the DOWNLOAD_URL
            haspic_col (str): df-schema column name of the
            HAS_PICTURE

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Downloading Gallery picture (HAS_PICTURE)")

            gc = self.gallery_controller
            gp = self.localg_path
            opt_in = self.get_wtags(args[1])
            haspic_data = gc.dlpaints(
                args[0],
                gp,
                opt_in[0],
                opt_in[1],
                opt_in[2],
                opt_in[3])

            ans = gc.updata(args[1], haspic_data)
            return ans
        # exception handling
        except Exception as exp:
            raise exp

    def eight(self, *args):
        """
        Option 8, based on the results of option 1, it scrap the search
        tags in each gallery element in the gallery

        Args:
            search_col (str): df-schema column name of SEARCH TAGS
            curl_col (str): df-schema column name of the COLLECTION
            vvg_url (str): web gallery root URL for the collection

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Recovering Gallery search tags (SEARCH_TAGS)")

            gc = self.gallery_controller
            opt_in = self.get_wtags(args[0])
            search_data = gc.scrap_searchtags(
                args[1],
                args[2],
                opt_in[0],
                opt_in[1],
                opt_in[2],
                opt_in[3],
                multiple=True)

            ans = gc.updata(args[0], search_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def nine(self, *args):
        """
        Option 9, based on the results of option 1, it scrap the
        object-data of each gallery element in the gallery

        Args:
            obj_col (str): df-schema column name of OBJ_DATA
            curl_col (str): df-schema column name of the COLLECTION

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Recovering Gallery object-data (OBJ_DATA)")

            gc = self.gallery_controller
            opt_in = self.get_wtags(args[0])
            object_data = gc.scrap_objdata(
                args[1],
                opt_in[0],
                opt_in[1],
                opt_in[2],
                multiple=False)

            ans = gc.updata(args[0], object_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def ten(self, *args):
        """
        Option 10, based on the results of option 1, it scrap the
        related work of each gallery element in the gallery

        Args:
            rwork_col (str): df-schema column name for RELATED_WORKS
            curl_col (str): df-schema column name of the COLLECTION
            vvg_url (str): web gallery URL search for the collection

        Raises:
            exp: raise a generic exception if something goes wrong

        Returns:
            ans (bool): boolean to confirm success of the task
        """
        try:
            print("Recovering Gallery related work (RELATED_WORKS)")

            gc = self.gallery_controller
            opt_in = self.get_wtags(args[0])
            rwork_data = gc.scrap_relwork(
                args[1],
                args[2],
                opt_in[0],
                opt_in[1],
                opt_in[2],
                opt_in[3],
                multiple=True)

            ans = gc.updata(args[0], rwork_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def eleven(self, *args):
        """
        Option 11, based on the results of option 1, it scrap the
        object-data of each gallery element in the gallery

        Args:
            id_col (str): df-schema column name of the ID
            img_col (str): df-schema column name of the IMG_DATA
            shape_col (str): df-schema column name of the IMG_SHAPE

        Raises:
            exp: raise a generic exception if something goes wrong
        """
        try:
            print("Exporting local images into RGB, B&W + shape")

            ans = False
            gc = self.gallery_controller
            ip = self.imgd_path

            opt_img = self.get_itags(args[1])
            opt_shape = self.get_itags(args[2])

            # create the local data/img folders
            print("Configuring local data image folders...")
            gc.create_localfolders(ip, args[0])

            # export the images in RGB and B&W
            print("Exporting RGB and B&W images...")
            img_data = gc.export_paints(args[0],
                                        opt_img[0],
                                        opt_img[1],
                                        opt_img[2])

            print("Getting RGB and B&W shapes...")
            shape_data = gc.export_shapes(args[0],
                                          opt_shape[0],
                                          opt_shape[1])

            # update the CSV columns with the data
            ans = gc.updata(args[1], img_data)
            ans = gc.updata(args[2], shape_data)
            return ans

        # exception handling
        except Exception as exp:
            raise exp

    def twelve(self, *args):
        """
        Option 12, export all scraped columns into JSON file in the
        designated local folders

        Raises:
            exp: raise a generic exception if something goes wrong
        """
        try:
            print("Exporting pandas-df to JSON in local gallery")

            gc = self.gallery_controller
            gp = self.localg_path

            # JSON export for the following columns:
            # - desccription
            # - search tags
            # - object date
            # - related work
            for temp_cname in args[1]:
                gc.export_json(
                    gp,
                    args[0],
                    temp_cname,
                    temp_cname.lower())

        # exception handling
        except Exception as exp:
            raise exp

    def printre(self, report):
        """
        prints the report tittle in the console

        Raises:
            exp: raise a generic exception if something goes wrong
        """
        try:
            print("=================== REPORT ===================")
            print("TASK COMPLETED: ", report)

        # exception handling
        except Exception as exp:
            raise exp

    def run(self):
        """
        Running the view class in the main

        Raises:
            exp: print the exception and restart the self.run() method
        """
        try:

            while True:
                self.menu()
                inp = self.inputs
                ans = False

                # known if the is auto or manual input
                if inp < 0:
                    inp = input("Select an option to continue\n")

                # starting a new gallery
                if int(inp) == 1:
                    ans = self.one(id_col, title_col, curl_col, vvg_url)

                # saving gallery in file
                elif int(inp) == 2:
                    ans = self.two(expf, dataf)

                # loading gallery in memory
                elif int(inp) == 3:
                    ans = self.three(expf, dataf, id_col)

                # checking gallery in memory
                elif int(inp) == 4:
                    self.four()
                    ans = True

                # recovering painting descriptions
                elif int(inp) == 5:
                    ans = self.five(desc_col, curl_col)

                # recovering painting download URL
                elif int(inp) == 6:
                    ans = self.six(dl_col, curl_col, vvg_url)

                # downloading painting file
                elif int(inp) == 7:
                    ans = self.seven(dl_col, haspic_col)

                # recovering painting search-tags
                elif int(inp) == 8:
                    ans = self.eight(search_col, curl_col, vvg_url)

                # recovering painting object-data
                elif int(inp) == 9:
                    ans = self.nine(obj_col, curl_col)

                # recovering painting related-work
                elif int(inp) == 10:
                    ans = self.ten(rwork_col, curl_col, vvg_url)

                # exporting and formating painting files
                elif int(inp) == 11:
                    ans = self.eleven(id_col, img_col, shape_col)

                # exporting CSV columns into JSON foles
                elif int(inp) == 12:
                    self.twelve(id_col, json_index_cols)
                    ans = True

                elif int(inp) == 99:
                    # list of automatic steps
                    # (3, 4, 5, 6, 7, 2, 8, 2, 9, 2, 10, 2, 11, 2, 12)
                    print("Auto executing options 5 to 12!!!...")
                    ans = self.five(desc_col, curl_col)
                    ans = self.two(expf, dataf)

                    ans = self.six(dl_col, curl_col, vvg_url)
                    ans = self.seven(dl_col, haspic_col)
                    ans = self.two(expf, dataf)

                    ans = self.eight(search_col, curl_col, vvg_url)
                    ans = self.two(expf, dataf)

                    ans = self.nine(obj_col, curl_col)
                    ans = self.two(expf, dataf)

                    ans = self.ten(rwork_col, curl_col, vvg_url)
                    ans = self.two(expf, dataf)

                    ans = self.eleven(id_col, img_col, shape_col)
                    ans = self.two(expf, dataf)

                    self.twelve(id_col, json_index_cols)
                    ans = self.two(expf, dataf)

                    self.inputs = -1

                # exit program
                elif int(inp) == 0:
                    sys.exit(0)

                # other option selected
                else:
                    print("Invalid option, please try again...")

                # printing report after finishing task
                self.printre(ans)

        # exception handling
        except Exception as exp:
            print(exp)
            self.run()


# main of the program
if __name__ == "__main__":
    # creating the View() object and running it
    scrapy = View()
    scrapy.setup()
    scrapy.run()
