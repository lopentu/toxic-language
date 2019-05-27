#!/bin/bash
OLDPWD=$(pwd)
cd $(dirname "$0")
source venv/bin/activate
nohup gunicorn server:app -b 0.0.0.0:15000 --workers 2 > server.log &
echo $! > server.pid
cd $OLDPWD

