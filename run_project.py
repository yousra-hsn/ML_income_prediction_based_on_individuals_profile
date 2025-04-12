import subprocess

def execute_files(file_list):
    for file in file_list:
        try:
            result = subprocess.run(['python', file], capture_output=True, text=True)
            print(f"Execution of {file}:\n{result.stdout}")
            if result.stderr:
                print(f"Errors in {file}:\n{result.stderr}")
        except Exception as e:
            print(f"An error occurred while executing {file}: {e}")

if __name__ == "__main__":
    print("Executing files...")
    files_to_execute = ['lib/lib_install.py', 'lib/lib_import.py', 'data/data_extract.py']
    execute_files(files_to_execute)