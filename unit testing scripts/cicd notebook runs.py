# Databricks notebook source
notebook_dict = {}

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

names = []
for i in w.workspace.list('/Shared/test-dbx-devops', recursive=True):
    notebook_dict[i.path] = i.modified_at
print(notebook_dict)

# COMMAND ----------

import json

# Define the path in DBFS
dbfs_path = '/dbfs/FileStore/tables/cicd_notebook_dict.json'

# Read the dictionary from a JSON file
with open(dbfs_path, 'r') as json_file:
    notebook_dict_old = json.load(json_file)
    print(notebook_dict_old)

# COMMAND ----------

# Find keys with changed values
changed_keys = []

# Check keys in the first dictionary
for key in notebook_dict_old:
    if key in notebook_dict and notebook_dict_old[key] != notebook_dict[key]:
        changed_keys.append(key)

# Check for keys only in the second dictionary
for key in notebook_dict:
    if key not in notebook_dict_old:
        changed_keys.append(key)

# Print the result
print("Keys with changed values:", changed_keys)

# COMMAND ----------

for i in changed_keys:
    dbutils.notebook.run(i, 0)

# COMMAND ----------

import json

# Define the path in DBFS
dbfs_path = '/dbfs/FileStore/tables/cicd_notebook_dict.json'

# Write the dictionary to a JSON file
with open(dbfs_path, 'w') as json_file:
    json.dump(notebook_dict, json_file)
