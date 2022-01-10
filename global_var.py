def prettyprint(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key), end='')
        if isinstance(value, dict):
            print('')
            prettyprint(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

ch2eng = {  
    "術語":"term" ,
    "連接":"connect" ,
    "分斷":"disconnection",
    "實地":"territory",
    "侵入":"invasion",
    "薄": "thin",
    "勢力":"influence",
    "攻擊":"attack",
    "防守":"defense",
    "好棋":"goodmove", 
    "壞棋":"badmove",
}

ch2abbr = {
    "術語":"gt" ,
    "連接":"ct" ,
    "分斷":"dt",
    "實地":"tr",
    "侵入":"in",
    "薄": "tn",
    "勢力":"if",
    "攻擊":"at",
    "防守":"df",
    "好棋":"gd", 
    "壞棋":"bd",
}


# ch2eng = {  
#             "手數":"cardinal", 
#             "黑棋":"bcolor", 
#             "白棋":"wcolor", 
#             "術語":"goterm", 
#             "變化圖":"branch", 
#             "盤面位置":"location",

#             "好棋":"goodmove", 
#             "壞棋":"badmove", 
#             "幾目":"territory", 
#             "勝率":"winrate", 
#             "劫爭": "ko",
            
#             "棋塊位置":"area", 
#             "棋塊狀態":"areastatus", 
#             "攻擊":"attack" ,
#             "防守":"defend", 
#             "眼位":"eyeinfo" , 
#             "死活":"lifedeath",
#             "區塊結果": "intent",

            
#             "全局分析": "boardstatus", 
#             "優勢":"good",
#             "劣勢":"bad", 
#             "難分類術語": "hardclassify",
# }

# ch2abbr = {
#     "手數":"cd", 
#     "黑棋":"bc", 
#     "白棋":"wc", 
#     "術語":"gt", 
#     "變化圖":"br", 
#     "盤面位置":"lc",
#     "好棋":"gm", 
#     "壞棋":"bm", 
#     "幾目":"tc", 
#     "勝率":"wr", 
#     "劫爭": "ko",
#     "棋塊位置":"ar", 
#     "棋塊狀態":"as", 
#     "攻擊":"at" ,
#     "防守":"df", 
#     "眼位":"ei" , 
#     "死活":"ld",
#     "區塊結果": "it",
#     "全局分析": "bs", 
#     "優勢":"gb",
#     "劣勢":"bb", 
#     "難分類術語": "hc",
# }