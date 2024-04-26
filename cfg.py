balance = ""
cfg = {
    "screen_x": 1280,
    "screen_y": 720,
    "min_detection_confidence": 0.65, 
    "min_tracking_confidence": 0.65, 
    "max_num_hands": 1, 
    "tolerance": 0.75, 
    "alpha": 0.75, 
    "btnClickDelay": 1.5, 
    "btnclr" : (175, 0, 175), 
    "txtclr" : (255,255,255),
    "btnparams": {
        "W": 400, 
        "H": 80, 
        "BtnSp": 20, 
        "R": 40, 
        "CirSp": 50 
    },
    "txtparams": {        
        "xadj": +20,
        "yadj": -20,        
        "font": 0,
        "fontScale": 1.8,
        "thickness": 2
    },
    "currentpage": "home",
    "curretpage": "Transactions", 
    "curpage": "Receipt",
    "pages": {
        "home" : {
            "pagetitle":["Welcome Our ATM", 100, 1.8, (175,0,175), 4],
            "buttons": ["","","Start","Exit","","","",""],
            "navigation": ["","","main","Exit","","","",""]
        },
            
        "main":{
            "pagetitle":["Select Match to Login", 100, 1.8, (175,0,175), 4],
            "buttons": ["","","","","","Match","",""],
            "navigation": ["","","","","","Match","",""]
        },

       
        "login":{           
            "pagetitle": ["Enter Password", 150, 2, (175,0,175), 4],            
            
            "buttons":["Keyboard", "", "", "", "", "" , "Exit", ""],
            "navigation": ["Keyboard", "", "",
                           "", "", "", "Exit", ""]
        },
        "Transactions":{
            "pagetitle": ["Select Transaction", 150, 2, (175,0,175), 4],
            
            "buttons":["Withdraw", "Balance", "Deposit", "", "", "Exit" , "", ""],
            "navigation": ["Withdraw", "Balance", "Deposit", "", "", "Exit", "", ""]
        },        

        
        "Receipt": {
            "pagetitle": ["Do you want a Receipt?", 150, 2, (175,0,175), 4],
            "buttons":["Yes", "", "No"],
            "navigation": ["WDDoneR", "", "WDDone"]
        },
        "WDDoneR": {
            "pagetitle": ["Please take your Card, Cash, & Receipt", 150, 1.8, (175,0,175), 4],
            "buttons":["", "", "", "", "New Txn", "", "", ""],
            "navigation": ["", "", "", "", "Transactions", "", "", ""]
        },
        "WDDone": {
            "pagetitle": ["Please take your Card & Cash", 150, 2, (175,0,175), 4],
            "buttons":["", "", "", "", "", "", "New Txn", ""],
            "navigation": ["", "", "", "", "", "", "Transactions", ""]
        }   
        
    }
}


