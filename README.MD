# Insurance Automation

Insurance automation system for customers using maching learning approches. This system includes Image processing and data analysing methods for follwing tasks.

- Vehicle detection
- Damage detection
- Claim prediction
- Churn prediction

## Initialize project

Install python 3.8 in pc. It comes with `pip` already installed.

1. Clone this Repo
1. Open folder in command line(Shell in linux) using `cd` command.
1. Run `pip install virtualenv`.
1. Run `virtualenv -p python .` (If `python3` is the command to run python 3.x then `virtualenv -p python3 .` )
1. Activate virtual environment using `Scripts\activate` (In linux `source bin/activate`)
1. `pip install -r requirements.txt`
1. Run `cd src`
1. Run `python manage.py makemigrations` and `python manage.py migrate` accordingly.
1. Download **DB Browser for SQLite** and install in computer.
1. Open db.sqlite3 file inside **DB Browser for SQLite**.
1. import data from _test.sql_ to db.sqlite3 and write changes to the database.
1. Download all the models from following link and extract into src\insurance\models

## Run server

Execute `python manage.py runserver` to run the server in local environment
