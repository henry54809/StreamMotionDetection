import configparser
import getopt
import os
import common
from pushbullet import Pushbullet

def init(raw_args):
     global args, cfg, pb
     optlist, _= getopt.getopt(raw_args, '', ['config_file='])
     args = dict(optlist)
     cfg = configparser.ConfigParser()
     cfg.read(args.get('--config_file', os.path.join(common.get_script_path(), 'default.cfg')))
     if cfg['NOTIFICATION'].get('PushbulletAPIKey', None):
         pb = Pushbullet(cfg['NOTIFICATION']['PushbulletAPIKey'])
