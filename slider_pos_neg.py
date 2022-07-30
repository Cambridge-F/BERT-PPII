#ecoding=utf-8
import os
import time
import glob
import csv

def splitByLineCount(filename,count):
    with open(filename, encoding='utf-8') as fin:
        i = True
        for line in fin:
            if i:
                buf = line.replace('\n','')
            else:
                flag = line.replace('\n','')
            i = False
        print('buf',buf,type(buf))
        print('flag',flag,type(flag))
        str_buf = ''
        str_flag = ''
        for j in range(int(count/2)):
            str_buf+='X'
            str_flag+='0'
        print("str",len(str_buf))
        buf = str_buf+buf+str_buf
        flag = str_flag+flag+str_flag
        buf = list(buf)
        flag = list(flag)
        WriteFiles(buf,flag,count)


def WriteFiles(buf,flag,count):
    print(type(buf), buf)
    print(type(flag), flag)
    print(count)
    for i in range(len(buf)-count+1):
        str_1 = "".join(buf[i:i + count:1])
        if int(flag[int((i+count+i)/2)])==1:
            print(buf[i:i+count:1])
            print(flag[i:i+count:1])
            print(str_1)
            print(len(str_1))
            #with open("F:\WS13\WS"+str(count)+"_pos_test.csv", 'a+') as f_pos:
            with open("./datasets/nonstrict_independence_data/WS9_independentTest/WS9_pos.csv", 'a+',encoding="utf-8", newline="") as f_pos:
                csv_writer_1 = csv.writer(f_pos)
                csv_writer_1.writerow([str_1,1])
                #f_pos.writelines(str_1)
        elif int(flag[int((i+count+i)/2)])==0:
            #with open("F:\WS13\WS"+str(count)+"_neg_test.csv","a+") as f_neg:
            with open("./datasets/nonstrict_independence_data/WS9_independentTest/WS9_neg.csv", "a+",encoding="utf-8", newline="") as f_neg:
                csv_writer_0 = csv.writer(f_neg)
                csv_writer_0.writerow([str_1,0])
                #f_neg.writelines(str_1)


if __name__ == '__main__':
    begin = time.time()
    i=0
    for fileName in glob.glob(os.path.join('./datasets/nonstrict_independence_data/Files_txt/', '*.{}'.format('txt'))):
        splitByLineCount(fileName, 9)#滑动窗口的大小
        i+=1
    print(i)
    end = time.time()
    print('time is %d seconds ' % (end - begin))