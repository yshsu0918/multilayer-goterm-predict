# Multilayer-goterm-predict 

## 流程

1. 看到盤面
2. model A -> 產生"術語分類" 
    - 這個盤面應該去描述 好棋 死活 優劣
    - 而非 攻擊 防守 
3. model B -> 決定各個"術語分類"中的術語 
    - 術語分類 "好棋, goodmove": 絕佳 尚可 普通 
    - 此盤面應該選絕佳
4. 利用model A的術語分類從template資料庫中尋找合適的template.
    - similarity
    - 
5. 把術語填回template (rule-base: CommentReplace.py)

## modelA 細節
board 19*19 -> BV EYE CONNECTOR + SCORE 
([ 19*19*6+2 * 1], [length of 分類])

好棋，攻擊，死活

[1,0,1]

## modelB 細節

training 
board 19*19 , 檢查該句子

實際使用


eng_abbrs = ['gt','gm','bm','at','df','ei','ld','gb','bb']


ch2abbr = {

    "術語":"gt",  
    "好棋":"gm", 
    "壞棋":"bm", 
    "攻擊":"at" ,
    "防守":"df", 
    "眼位":"ei" , 
    "優勢":"gb",
    "死活":"ld"
    "劣勢":"bb", 
}

gt ['走', '角', '點', '着', '扳', '一手', '打', '斷', '空', '粘', '劫材', '劫', '接']
gm ['大', '必然', '好手', '嚴厲']
bm ['不好', '難受', '不利', '貪']
at ['攻擊', '作戰', '反擊']
df ['聯絡', '做活', '加強']
ei ['眼', '眼位']
ld ['安定', '活', '死']
gb ['優', '優勢', '領先']
bb ['差', '苦', '劣', '不行']



## New dataset establish 建立dataset
cd /mnt/nfs/work/yshsu0918/lal/trunk
./podman.sh
vim CGI/WeichiCommonGtpEngine.cpp
comment_feature function 中有描述input檔案路徑
修改完畢保存
./runcgi.sh 
    輸入 comment_feature

產生 .bv_eye_connect

利用 gen_sentence_order_tags_trainingdata
產生 trainning.pickle

## 


## 多個模型比較performance

1. 每種術語分別訓練
2. 術語類別A
3. 術語類別B

## 實作功能

1. 棋譜+評論 (可參考江凡的visualize.py)
2. 或是 用網頁的形式
3. template
    - template 尋找器
    - 需要斷詞嗎? 兩種都試試看





## Appendix


### hans30805
N=bcolor
N=wcolor
N=go_term_N
N=go_ko_N
N=go_ladder
N=go_liberty_N
N=go_territory
N=go_liveordead_N
N=go_thin
N=go_thick
N=go_terms
N=go_board_state_N
N=go_opening
N=go_influence
N=go_planning
N=go_variation
N=go_hardclassified_N
N=winrate
N=area
N=areastatus
N=boardstatus
N=territory
Vt=go_term_V
Vt=go_ko_V
Vt=go_connect
Vt=go_attack
Vt=go_defense
Vt=go_liberty_V
Vt=go_invasion
Vt=go_erasure
Vt=go_board_state_V
Vt=go_sente
Vt=go_hardclassified_V
ADJ=go_liveordead_ADJ
ADJ=goodmove
ADJ=badmove
territory=NUM:目
winrate=NUM:%
NP=go:NUM
NP=ALPHABET:位
bcolor	黑棋 黑 黑子
wcolor	白棋 白 白子
go_term_N	目外 高目 三三 五五 天元 小目 星位
go_term_V	三三 倒撲 渡 沖 接 掛 貼 黏 託 覰 締 凹 崩 逼 並 拆 拆逼 長 吃 衝 冲 打 打吃 渡 斷 尖 飛 頂 封 虎 扳 逼 拆 拆逼 壓 雙 圍 擋 點 调 刺 立 粘 扭斷 碰 跨斷 雙吃 雙打 雙打吃 雙倒撲 雙掛 跳 門吃 擠 夾 靠 扳斷 反扳 連扳 尖衝 象飛 尖頂 扭斷 點角 掛角 雙叫吃 雙飛燕 叫吃 門 擋 拐 扳斷 頂 大飛 小飛 飛掛 飛壓 退 托 挖 尖衝 尖頂 爬 跨 反提 反扳 反打吃 反夾 撲 覷 提 提子 提吃 吃掉 締角 飛壓 立下 沖斷 鎮 曲 小飛 小尖 象飛 穿象眼 大飛 做虎口 託斷 尖斷 雙虎 拆邊 二間拆 二間跳 拆二 拆一 二五侵分 二四侵分 鎮頭 夾攻 虛 鼻頂 衝出 壓出 靠壓 單壓
go_ko_N	劫 劫材 征子劫 劫材 劫爭 劫材 萬劫不應 循環劫 損劫 寬氣劫 消劫 無憂劫 萬年劫 初棋無劫 先手劫 雙劫 劫活 鬆氣劫 單劫 三劫 三劫循環 套劫 劫殺 劫争 本身劫 緩氣劫 後手劫 天下大劫 生死劫 緊氣劫 連環劫 賴皮劫 二手劫
go_ko_V	打劫 開劫 提劫 做劫
go_ladder	征子 引征 征

go_connect	連回 連 連接 分斷 切斷 斷點 阻渡 渡過 聯絡 接上

go_attack	纏繞 壓迫 追擊 殺 攻 攻擊 殺棋 淨殺 對殺 戰鬥 激戰 想全殲 反攻 反擊 追擊 封鎖

go_defense	騰挪 逃 逃遁 逃生 抵抗 防禦 防守 做活 加強 退讓 補強

go_liberty_N	氣 公氣 寬氣 寬一氣 外氣 內氣 不入 不入子 禁著點 氣合點 雙活
go_liberty_V	緊氣 收氣 長氣 比氣



go_territory	實地 實利 實地 根據地 地域 地 地盤 盤面目數 陣地 目數
go_invasion	打入 侵入
go_erasure	淺消 浅消 削減 壓縮 侵消 壓縮 削減
go_liveordead_N	自殺 做活 死活 兩眼 真眼 有眼殺瞎 眼 眼位 眼形 假眼 假眼活 求活 死棋 死子 活棋 活子 活型 死形 活形 一隻眼 共活 
go_liveordead_ADJ	死 活    
go_thin	孤棋 弱棋 餘味 味道 全盤薄形
go_thick	補厚 厚勢 厚實 補強 補 自補 加固 牆 牆壁
go_terms	接不歸 無憂角 脹牯牛 脹死牛 閃電四 盤角板六 盤角曲四 大豬嘴 大頭鬼 小豬嘴 金櫃角 板六 刀把五 刀板五 刀五 龜甲 開花 梅花六 梅花五 彎三 彎四 金雞獨立 曲三 曲四 蒼鷹搏兔 黃鶯撲蝶 老鼠偷油 方四 愚形 大壓梁 直二 直三 直四 兩頭蛇 空三角 緊帶鈎 烏龜不出頭 扭十字 扭十字長一方
go_board_state_N	開局 分投 中盤 官子 單官 大伸腿 佈局 逆官子 小伸腿
go_board_state_V	收官 收官子 
go_opening	佈局 布局 三連星 二連星 秀策流 中國流 低中國流 小林流 三連星 宇宙流
go_influence	勢力 大模樣 模樣 大空 外勢
go_planning	雪崩 妖刀 倚蓋 倒垂蓮 定式 台象
go_sente	脫先 先手 逆先 逆先手 後手
go_variation	實戰 實戰手 變化 變化圖
go_hardclassified_N	戀子 進去 救 整形 孤立 本手 拆逼 大盤 長生 次序 大場 大局感 大龍 點眼 定型 浮棋 浮子 互圍 回提 見合 交叉點 交點 龍 盲點 盤渡 騙著 氣緊 伸氣 試應手 守角 手筋 鐵柱 透點 問應手 細棋 虛手 虛著 要點 要子 疑問手 硬腿 裝倒撲 俗手 兩分 細棋 虎口 手筋 取捨 定型 大龍 本形 二子頭必扳
go_hardclassified_V	擴大 擴張 擴展 逆轉 保留 包圍 滾打 交換 轉換 治孤 滾打吃 調子 衝擊 還原 破眼 壯大 突圍 出頭 互破 淨吃 利用
goodmove	形勢不錯 好一些 優勢 領先 小勝 稍好 輕鬆 大利 佔優 好手 起作用 攻守兼備 恰到好處 最佳應手 一手便宜 當然的 肯定的 好次序 好手段N=bcolor
N=wcolor
N=go_term_N
N=go_ko_N
N=go_ladder
N=go_liberty_N
N=go_territory
N=go_liveordead_N
N=go_thin
N=go_thick
N=go_terms
N=go_board_state_N
N=go_opening
N=go_influence
N=go_planning
N=go_variation
N=go_hardclassified_N
N=winrate
N=area
N=areastatus
N=boardstatus
N=territory
Vt=go_term_V
Vt=go_ko_V
Vt=go_connect
Vt=go_attack
Vt=go_defense
Vt=go_liberty_V
Vt=go_invasion
Vt=go_erasure
Vt=go_board_state_V
Vt=go_sente
Vt=go_hardclassified_V
ADJ=go_liveordead_ADJ
ADJ=goodmove
ADJ=badmove
territory=NUM:目
winrate=NUM:%
NP=go:NUM
NP=ALPHABET:位
bcolor	黑棋 黑 黑子
wcolor	白棋 白 白子
go_term_N	目外 高目 三三 五五 天元 小目 星位
go_term_V	三三 倒撲 渡 沖 接 掛 貼 黏 託 覰 締 凹 崩 逼 並 拆 拆逼 長 吃 衝 冲 打 打吃 渡 斷 尖 飛 頂 封 虎 扳 逼 拆 拆逼 壓 雙 圍 擋 點 调 刺 立 粘 扭斷 碰 跨斷 雙吃 雙打 雙打吃 雙倒撲 雙掛 跳 門吃 擠 夾 靠 扳斷 反扳 連扳 尖衝 象飛 尖頂 扭斷 點角 掛角 雙叫吃 雙飛燕 叫吃 門 擋 拐 扳斷 頂 大飛 小飛 飛掛 飛壓 退 托 挖 尖衝 尖頂 爬 跨 反提 反扳 反打吃 反夾 撲 覷 提 提子 提吃 吃掉 締角 飛壓 立下 沖斷 鎮 曲 小飛 小尖 象飛 穿象眼 大飛 做虎口 託斷 尖斷 雙虎 拆邊 二間拆 二間跳 拆二 拆一 二五侵分 二四侵分 鎮頭 夾攻 虛 鼻頂 衝出 壓出 靠壓 單壓
go_ko_N	劫 劫材 征子劫 劫材 劫爭 劫材 萬劫不應 循環劫 損劫 寬氣劫 消劫 無憂劫 萬年劫 初棋無劫 先手劫 雙劫 劫活 鬆氣劫 單劫 三劫 三劫循環 套劫 劫殺 劫争 本身劫 緩氣劫 後手劫 天下大劫 生死劫 緊氣劫 連環劫 賴皮劫 二手劫
go_ko_V	打劫 開劫 提劫 做劫
go_ladder	征子 引征 征

go_connect	連回 連 連接 分斷 切斷 斷點 阻渡 渡過 聯絡 接上

go_attack	纏繞 壓迫 追擊 殺 攻 攻擊 殺棋 淨殺 對殺 戰鬥 激戰 想全殲 反攻 反擊 追擊 封鎖

go_defense	騰挪 逃 逃遁 逃生 抵抗 防禦 防守 做活 加強 退讓 補強

go_liberty_N	氣 公氣 寬氣 寬一氣 外氣 內氣 不入 不入子 禁著點 氣合點 雙活
go_liberty_V	緊氣 收氣 長氣 比氣

go_territory	實地 實利 實地 根據地 地域 地 地盤 盤面目數 陣地 目數
go_invasion	打入 侵入
go_erasure	淺消 浅消 削減 壓縮 侵消 壓縮 削減
go_liveordead_N	自殺 做活 死活 兩眼 真眼 有眼殺瞎 眼 眼位 眼形 假眼 假眼活 求活 死棋 死子 活棋 活子 活型 死形 活形 一隻眼 共活 
go_liveordead_ADJ	死 活
go_thin	孤棋 弱棋 餘味 味道 全盤薄形
go_thick	補厚 厚勢 厚實 補強 補 自補 加固 牆 牆壁
go_terms	接不歸 無憂角 脹牯牛 脹死牛 閃電四 盤角板六 盤角曲四 大豬嘴 大頭鬼 小豬嘴 金櫃角 板六 刀把五 刀板五 刀五 龜甲 開花 梅花六 梅花五 彎三 彎四 金雞獨立 曲三 曲四 蒼鷹搏兔 黃鶯撲蝶 老鼠偷油 方四 愚形 大壓梁 直二 直三 直四 兩頭蛇 空三角 緊帶鈎 烏龜不出頭 扭十字 扭十字長一方
go_board_state_N	開局 分投 中盤 官子 單官 大伸腿 佈局 逆官子 小伸腿
go_board_state_V	收官 收官子 
go_opening	佈局 布局 三連星 二連星 秀策流 中國流 低中國流 小林流 三連星 宇宙流
go_influence	勢力 大模樣 模樣 大空 外勢
go_planning	雪崩 妖刀 倚蓋 倒垂蓮 定式 台象
go_sente	脫先 先手 逆先 逆先手 後手
go_variation	實戰 實戰手 變化 變化圖
go_hardclassified_N	戀子 進去 救 整形 孤立 本手 拆逼 大盤 長生 次序 大場 大局感 大龍 點眼 定型 浮棋 浮子 互圍 回提 見合 交叉點 交點 龍 盲點 盤渡 騙著 氣緊 伸氣 試應手 守角 手筋 鐵柱 透點 問應手 細棋 虛手 虛著 要點 要子 疑問手 硬腿 裝倒撲 俗手 兩分 細棋 虎口 手筋 取捨 定型 大龍 本形 二子頭必扳
go_hardclassified_V	擴大 擴張 擴展 逆轉 保留 包圍 滾打 交換 轉換 治孤 滾打吃 調子 衝擊 還原 破眼 壯大 突圍 出頭 互破 淨吃 利用
goodmove	形勢不錯 好一些 優勢 領先 小勝 稍好 輕鬆 大利 佔優 好手 起作用 攻守兼備 恰到好處 最佳應手 一手便宜 當然的 肯定的 好次序 好手段 有力的 穩定 有利 急所 絕佳 必然 好棋 嚴厲 好點 厲害 穩健 輕靈 不壞 好調 巧妙 合算 好形 大
badmove	優勢喪失殆盡 越來越糟 難於應付 損失太大 難走一些 左右爲難 遭透了 不完備 難受 空不夠 劣勢 困難 難辦 不行 單調 有點問題 稍感急躁 兩手不當 胸無成算 問題手 不太好 無理的 不成立 太急躁 行不通 非常壞 躁進 不利 不好 艱辛 難受 錯誤 壞棋 無理 俗手 軟弱 敗着 惋惜 惡手 貪 緩 壞手
area	左上角 右下角 左下角 右上角 一塊棋 兩塊棋 上面 下面 左邊 右邊 中間 中腹 中央 黑陣 右下 右上 左上 左下 中下 上邊 角部 二子 外面 兩塊 角上 下邊 邊上
areastatus	無條件死亡 死活問題 安頓不好 富有彈性 輕靈點 接成棒 殺不盡 不舒服 實地 模樣 定型 激戰 戰鬥 孤立 崩潰 不安 轉換 新型 大陣 厚實 威脅 豐厚 擴展 交換 大龍 大空 難受 薄味 厚勢 愚形 壯大  變壞 急所 重複 薄形 惡味 充分 本形 疑問 膨脹 堅實 重 厚 弱 薄
boardstatus	全局形勢 全盤薄形 快步調 勝勢 局勢 勝負 局面 時機 外勢 本局 兩分 細棋 細 波折
有力的 穩定 有利 急所 絕佳 必然 好棋 嚴厲 好點 厲害 穩健 輕靈 不壞 好調 巧妙 合算 好形 大
badmove	優勢喪失殆盡 越來越糟 難於應付 損失太大 難走一些 左右爲難 遭透了 不完備 難受 空不夠 劣勢 困難 難辦 不行 單調 有點問題 稍感急躁 兩手不當 胸無成算 問題手 不太好 無理的 不成立 太急躁 行不通 非常壞 躁進 不利 不好 艱辛 難受 錯誤 壞棋 無理 俗手 軟弱 敗着 惋惜 惡手 貪 緩 壞手
area	左上角 右下角 左下角 右上角 一塊棋 兩塊棋 上面 下面 左邊 右邊 中間 中腹 中央 黑陣 右下 右上 左上 左下 中下 上邊 角部 二子 外面 兩塊 角上 下邊 邊上
areastatus	無條件死亡 死活問題 安頓不好 富有彈性 輕靈點 接成棒 殺不盡 不舒服 實地 模樣 定型 激戰 戰鬥 孤立 崩潰 不安 轉換 新型 大陣 厚實 威脅 豐厚 擴展 交換 大龍 大空 難受 薄味 厚勢 愚形 壯大  變壞 急所 重複 薄形 惡味 充分 本形 疑問 膨脹 堅實 重 厚 弱 薄
boardstatus	全局形勢 全盤薄形 快步調 勝勢 局勢 勝負 局面 時機 外勢 本局 兩分 細棋 細 波折
