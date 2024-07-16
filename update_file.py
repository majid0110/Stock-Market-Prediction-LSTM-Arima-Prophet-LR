import os
from github import Github

def update_file():
    g = Github(os.environ['GH_PAT'])
    
    repo = g.get_repo("majid0110/Stock-Market-Prediction-LSTM-Arima-Prophet-LR")
    
    file_path = "daily_updates.py"
    new_content = "print('hello')\n"

    try:
        # Try to get the file contents
        file = repo.get_contents(file_path)
        
        # If the file exists, update it
        repo.update_file(
            path=file_path,
            message="Daily update",
            content=new_content,
            sha=file.sha
        )
        print(f"File {file_path} updated successfully")
    except Exception as e:
        if "Not Found" in str(e):
            # If the file doesn't exist, create it
            repo.create_file(
                path=file_path,
                message="Create daily_updates.py",
                content=new_content
            )
            print(f"File {file_path} created successfully")
        else:
            # If there's another error (like SHA mismatch), fetch the latest content and update
            latest_file = repo.get_contents(file_path)
            repo.update_file(
                path=file_path,
                message="Daily update",
                content=new_content,
                sha=latest_file.sha
            )
            print(f"File {file_path} updated successfully after resolving conflicts")

if __name__ == "__main__":
    update_file()
