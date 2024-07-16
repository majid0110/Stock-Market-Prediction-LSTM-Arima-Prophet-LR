import os
from github import Github

def update_file():
    g = Github(os.environ['GH_PAT'])
    
    repo = g.get_repo("majid0110/Stock-Market-Prediction-LSTM-Arima-Prophet-LR")
    
    file = repo.get_contents("stock_market_prediction.py")
    
    new_content = "print('hello')\n"
    
    repo.update_file(
        path="stock_market_prediction.py",
        message="Daily update",
        content=new_content,
        sha=file.sha
    )

if __name__ == "__main__":
    update_file()
