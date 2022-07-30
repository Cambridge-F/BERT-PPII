#将.dataset格式的文件 拆分成每一条是一个文件
#ecoding=utf-8
def split_dataset(fileName):
    L = []
    with open(fileName,encoding="utf-8",newline="") as f:
        for i in f:
            L.append(i)
            if len(L) == 3:
                sub_fileName = L[0]
                sub_fileName = sub_fileName.replace('\n','').replace('\r','')
                with open("./datasets/nonstrict_independence_data/Files_txt/" + sub_fileName+".txt", 'w') as fout:
                    fout.write(L[1])
                    fout.write(L[2])
                L.clear()
if __name__ == '__main__':
    split_dataset("./datasets/nonstrict_independence_data/independentTestSeqFeatureMappings.dataset")
