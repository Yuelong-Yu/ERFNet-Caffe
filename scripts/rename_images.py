import os

class BatchRename():
    def __init__(self):
        self.path = '/media/yly/AIvehicle/CITYSCAPES_DATASET/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_02_copy'

    def rename(self):
        filelist = os.listdir(self.path)
        filelist= sorted(filelist)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i), '0>4s') + '.png')
                try:
                    os.rename(src, dst)
                    print 'converting %s to %s ...' % (src, dst)
                    i = i + 1
                except:
                    continue
        print 'total %d to rename & converted %d jpgs' % (total_num, i)

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()