import os
from github import Github

def update_file():
    print("Starting update_file function")
    g = Github(os.environ['GH_PAT'])
    
    repo = g.get_repo("majid0110/Stock-Market-Prediction-LSTM-Arima-Prophet-LR")
    print(f"Repository accessed: {repo.full_name}")
    
    file_path = "daily_updates.py"
    new_content = "print('hello')\n"

    try:
        print(f"Attempting to get contents of {file_path}")
        file = repo.get_contents(file_path)
        print(f"File {file_path} found. SHA: {file.sha}")
        
        print(f"Updating file {file_path}")
        repo.update_file(
            path=file_path,
            message="Daily update",
            content=new_content,
            sha=file.sha
        )
        print(f"File {file_path} updated successfully")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        if "Not Found" in str(e):
            print(f"File {file_path} not found. Creating it.")
            repo.create_file(
                path=file_path,
                message=f"Create {file_path}",
                content=new_content
            )
            print(f"File {file_path} created successfully")
        else:
            print("Attempting to resolve conflict")
            latest_file = repo.get_contents(file_path)
            print(f"Latest file SHA: {latest_file.sha}")
            repo.update_file(
                path=file_path,
                message="Daily update",
                content=new_content,
                sha=latest_file.sha
            )
            print(f"File {file_path} updated successfully after resolving conflicts")

if __name__ == "__main__":
    update_file()
