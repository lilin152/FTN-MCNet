"""Create synthetic, non-clinical images and matching .npy / CSV indices.

This checks index parsing and file layout only; clinical performance is meaningless.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image


def main():
    root=Path(__file__).resolve().parent/'demo_data'
    split_dir=root/'splits'
    split_dir.mkdir(parents=True,exist_ok=True)
    mapping={'train':[('demo_train_01',0),('demo_train_02',1),('demo_train_03',2),('demo_train_04',3)],
             'valid':[('demo_valid_01',0),('demo_valid_02',1),('demo_valid_03',2),('demo_valid_04',3)],
             'test_external1':[('demo_test1_01',0),('demo_test1_02',1),('demo_test1_03',2),('demo_test1_04',3)],
             'test_external2':[('demo_test2_01',0),('demo_test2_02',1),('demo_test2_03',2),('demo_test2_04',3)]}
    with (root/'reports.csv').open('w',newline='',encoding='utf8') as fh:
        writer=csv.writer(fh);writer.writerow(['patient_id','report_text'])
        for split,patient_list in mapping.items():
            rows=[]
            for k,(pid,label) in enumerate(patient_list):
                folder=root/'images'/pid
                folder.mkdir(parents=True,exist_ok=True)
                pixels=np.random.default_rng(k+label*47+len(split)).integers(10,240,size=(128,160),dtype=np.uint8)
                Image.fromarray(pixels).save(folder/'image_01.png')
                rows.append([str(folder),label,pid])
                writer.writerow([pid,f'Synthetic ultrasound description, not a clinical record.'])
            np.save(split_dir/f'{split}.npy',np.asarray(rows,dtype=object))
    print('Synthetic data ready:',root)
    print('python train.py --data-dir',split_dir,'--report-csv',root/'reports.csv','--dry-run')

if __name__=='__main__': main()
