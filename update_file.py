import os
from github import Github

def update_file():
    g = Github(os.environ['GH_PAT'])
    
    repo = g.get_repo("majid0110/Stock-Market-Prediction-LSTM-Arima-Prophet-LR")
    
    daily_updates = repo.get_contents("daily_updates.py")
    
    try:
        file = repo.get_contents("stock_market_prediction.py")
        new_content = "print('hello')\n"
        
        repo.update_file(
            path="stock_market_prediction.py",
            message="Daily update",
            content=new_content,
            sha=file.sha  # Use the SHA of stock_market_prediction.py
        )
        print("File updated successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # If the file doesn't exist, create it
        if "Not Found" in str(e):
            repo.create_file(
                path="stock_market_prediction.py",
                message="Create stock_market_prediction.py",
                content="print('hello')\n"
            )
            print("File created successfully")

if __name__ == "__main__":
    update_file()
