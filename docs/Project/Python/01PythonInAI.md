The venv to set up the virtual env.

pip install ipykernel

Jupyter > Interactive Window > Text editor: Execute Selector
We will be able to run the selected code and not the entire file.

Storing values - List, dictionary, Tuple and Set.
```py
my_list = ["Alice",25,True];

my_dict = {"name":"Alice","age":289}

print(my_dict["name"])

my_tuple = (red, green)

my_set = {1,2,3,4,5}
set_value = set(my_set)
```
The API key and all value should be in env variable and in os not in the code.
```py
import os
api_key = os.environ.get("API_Key")
database = os.environ.get("Database_name")
print(f"The database - {database}")
```
The values are None. 
The terminal export the value- `export API_Key=apiKey123` in bash and in window - `$env:API_KEY = "apikey123"`

```
(.venv) PS K:\Programming\Python_AI> $env:API_KEY = "apikey123"
(.venv) PS K:\Programming\Python_AI> echo $env:API_KEY
apikey123
```

In real time the export run everytime to get teh value. The most easy way to make a .env file and in gitignore remove the file and add the value in the file.
In terminal install `pip install python-dotenv`.

In the file the apikey is used `from dotenv import load_dotenv` and then the load_dotenv() it will get the data form the env file.


In styling the code extension Ruff and make the default formatter set up and the format after save. It will make the code format in the python file.