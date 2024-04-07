from flask import Flask, render_template, request, redirect, url_for, session
import torch
import os
from PIL import Image
import io
import collections
from uuid import uuid4
import json
import argparse
import glob
from ast import literal_eval
import numpy as np
import sys


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


app.run()
