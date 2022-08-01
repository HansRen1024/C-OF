# -*-coding:utf-8-*-
from facenet_pytorch import MTCNN, RNet
from PIL import Image, ImageDraw
import torch, mmcv, cv2, time, json, os
import numpy as np
from torch.nn.functional import interpolate

def test_mtcnn_img():
    mtcnn = MTCNN(image_size=640, thresholds=[0.8, 0.8, 0.6], min_face_size=40)
    img = Image.open('./test_1.jpg')
    boxes, _ = mtcnn.detect(img)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    img.show()

def test_mtcnn_video():
    device = torch.device('cuda')
    mtcnn = MTCNN(image_size=320, thresholds=[0.8, 0.8, 0.6], min_face_size=100, device=device)
    video = mmcv.VideoReader('video2.mp4')
    mean_ms = []
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('queen-mtcnn.avi', fourcc, 60.0, (1280, 720))
    dat = dict(dir_name="queen", bboxes=[])
    for frame in video:
        T1 = time.time()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)
        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is None:
            dat['bboxes'].append([0, 0, 0, 0])
        else:
            for bb in boxes:
                dat['bboxes'].append([int(bb[0]), int(bb[1]), int(bb[2] - bb[0]), int(bb[3] - bb[1])])
                draw.rectangle(bb.tolist(), outline=(255, 0, 0), width=6)
        frame_draw = cv2.cvtColor(np.asarray(frame_draw),cv2.COLOR_RGB2BGR)
        T2 = time.time()
        mean_ms.append(round((T2 - T1)*1000, 4))
        # FPS = round(1000/np.mean(mean_ms), 4)
        frame_draw = cv2.putText(frame_draw, str(np.round(np.mean(mean_ms), 4))+"ms", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # frame_draw = cv2.putText(frame_draw, str(FPS) + "FPS", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # out.write(frame_draw)
        cv2.imshow("MTCNN", frame_draw)
        k = cv2.waitKey(1)
        if k ==27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    # out.release()
    # file_out = 'results/queen/MTCNN.json'
    # with open(file_out, 'w', encoding='utf-8') as fout:
    #     fout.write(json.dumps(dat))
    #     fout.write('\n')

class box:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
    def clear(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
    def scale(self, scale):
        self.x = self.x/scale
        self.y = self.y/scale
        self.w = self.w/scale
        self.h = self.h/scale

class optical_flow_tracking:
    def __init__(self):
        self.last_box = box()
        self.points1 = None
        self.points2 = None
        self.pointsFB = None
        self.tbb = box()
        self.tracked = False
        term_criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        self.lk_params = dict(winSize=(4,4),maxLevel=5,criteria=term_criteria)

    def get_last_box(self, cbox: box()):
        self.last_box = cbox

    def process_frame(self, last_gray,current_gray, bbnext, lastboxfound):
        self.points1 = None
        self.points2 = None
        if lastboxfound:
            self.track(last_gray,current_gray)
        else:
            self.tracked = False
        if self.tracked:
            bbnext = self.tbb
            self.last_box = bbnext
        else:
            lastboxfound = False
        return bbnext, lastboxfound

    def track(self, last_gray, current_gray):
        self.bbPoints()
        if len(self.points1)<1:
            self.tracked = False
            return
        self.tracked = self.trackf2f(last_gray, current_gray)
        if self.tracked:
            self.bbPredict()

    def bbPoints(self):
        max_pts = 10
        margin_h = 0
        margin_v = 0
        step_x = int((self.last_box.w - 2*margin_h)/max_pts)
        step_y = int((self.last_box.h - 2*margin_v)/max_pts)
        points = []
        for y in range(self.last_box.y+margin_v, self.last_box.y+self.last_box.h-margin_v, step_y):
            for x in range(self.last_box.x+margin_h, self.last_box.x+self.last_box.w-margin_h, step_x):
                points.append((float(x), float(y)))
        self.points1 = np.expand_dims(np.array(points).astype(np.float32),1)

    def trackf2f(self, last_gray, current_gray):
        self.points2, status, similarity = cv2.calcOpticalFlowPyrLK(last_gray, current_gray,
                                                                     self.points1, None,
                                                                     **self.lk_params)
        self.pointsFB, FBstatus, FBerror = cv2.calcOpticalFlowPyrLK(current_gray, last_gray,
                                                                    self.points2, None,
                                                                    **self.lk_params)
        for idx in range(self.points1.shape[0]):
            real = self.pointsFB[idx][0][0] - self.points1[idx][0][0]
            imag = self.pointsFB[idx][0][1] - self.points1[idx][0][1]
            FBerror[idx] = pow(real, 2) + pow(imag, 2)
        status, similarity = self.normCrossCorrelation(last_gray, current_gray, status, similarity)
        return self.filterPts(status, similarity, FBerror)

    def normCrossCorrelation(self, last_gray, current_gray, status, similarity):
        for idx in range(self.points1.shape[0]):
            if status[idx]==1:
                rec0 = cv2.getRectSubPix(last_gray, (10,10), (int(self.points1[idx][0][0]), int(self.points1[idx][0][1])))
                rec1 = cv2.getRectSubPix(current_gray, (10,10), (int(self.points2[idx][0][0]), int(self.points2[idx][0][1])))
                res = cv2.matchTemplate(rec0, rec1, cv2.TM_CCOEFF_NORMED)
                similarity[idx] = float(res[0])
            else:
                similarity[idx] = float(0)
        return status, similarity

    def filterPts(self, status, similarity, FBerror):
        simmed = np.median(similarity)
        k = 0
        for i in range(self.points2.shape[0]):
            if not status[i]: continue
            if similarity[i] >= simmed:
                self.points1[k] = self.points1[i]
                self.points2[k] = self.points2[i]
                FBerror[k] = FBerror[i]
                k+=1
        if k == 0: return False
        self.points1 = self.points1[:k]
        self.points2 = self.points2[:k]
        FBerror = FBerror[:k]
        fbmed = np.median(FBerror)
        k = 0
        for i in range(self.points2.shape[0]):
            if not status[i]: continue
            if FBerror[i] <= fbmed:
                self.points1[k] = self.points1[i]
                self.points2[k] = self.points2[i]
                k+=1
        self.points1 = self.points1[:k]
        self.points2 = self.points2[:k]
        if k > 0: return True
        else: return False

    def bbPredict(self):
        npoints = self.points1.shape[0]
        xoff = np.zeros(npoints)
        yoff = np.zeros(npoints)
        for i in range(npoints):
            xoff[i] = self.points2[i][0][0] - self.points1[i][0][0]
            yoff[i] = self.points2[i][0][1] - self.points1[i][0][1]
        dx = np.median(xoff)
        dy = np.median(yoff)
        if npoints>1:
            d = []
            for i in range(npoints):
                for j in range(i+1, npoints):
                    d.append(np.sqrt(pow(self.points2[i][0][0] - self.points2[j][0][0], 2) + pow(self.points2[i][0][1] - self.points2[j][0][1], 2))/
                             np.sqrt(pow(self.points1[i][0][0] - self.points1[j][0][0], 2) + pow(self.points1[i][0][1] - self.points1[j][0][1], 2)))
            s = np.median(d)
        else:
            s = 1.0
        s1 = 0.5 * (s - 1) * self.last_box.w
        s2 = 0.5 * (s - 1) * self.last_box.h
        self.tbb.x = max(int(np.round(self.last_box.x + dx -s1)), 0)
        self.tbb.y = max(int(np.round(self.last_box.y + dy - s2)), 0)
        self.tbb.w = int(np.round(self.last_box.w * s))
        self.tbb.h = int(np.round(self.last_box.h * s))

class cof:
    def __init__(self, device, image_size=640):
        # device = torch.device('cuda:0')
        self.image_size=image_size
        self.mtcnn = MTCNN(image_size=image_size, thresholds=[0.8, 0.8, 0.6], min_face_size=40, device=device)
        self.rnet = RNet()
        self.final_box = []
        self.pbox = box()
        self.cbox = box()
        self.status = True
        self.tracking = optical_flow_tracking()
        self.skip = 5
        self.last_gray = None
        self.result = box()
        self.t0_list = []
        self.t1_list = []
        self.t2_list = []

    def detect(self, img):
        if len(self.final_box)==0 or self.status==False:
            t0 = time.time()
            self.result.clear()
            with torch.no_grad():
                self.final_box, _ = self.mtcnn.detect(img)
            self.t0_list.append(np.round((time.time()- t0)*1000, 4))
            # print(f'>>>> mtcnn time: {np.round(np.mean(self.t0_list), 4)} ms')
            if self.final_box is None:
                self.final_box = []
            if len(self.final_box)>0:
                self.cbox.x = int(self.final_box[0][0])
                self.cbox.y = int(self.final_box[0][1])
                self.cbox.w = int(self.final_box[0][2] - self.final_box[0][0])
                self.cbox.h = int(self.final_box[0][3] - self.final_box[0][1])
                self.last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.tracking.get_last_box(self.cbox)
                self.result = self.cbox
                self.status = True
        if len(self.final_box) > 0:
            current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            self.pbox, self.status = self.tracking.process_frame(self.last_gray,current_gray,self.pbox, self.status)
            self.t1_list.append(np.round((time.time() - t1) * 1000, 4))
            # print(f'tracking time: {np.round(np.mean(self.t1_list), 4)} ms')
            if self.status:
                if self.skip>2:
                    t2 = time.time()
                    face = img[self.pbox.y:self.pbox.y+self.pbox.h, self.pbox.x:self.pbox.x+self.pbox.w]
                    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face = np.uint8(face)
                    rnet_in = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float()
                    rnet_in = interpolate(rnet_in, (24, 24))
                    rnet_in = (rnet_in - 127.5) * 0.0078125
                    with torch.no_grad():
                        out = self.rnet(rnet_in)
                    out1 = out[1].permute(1, 0)
                    score = out1[1, :]
                    if score<0.90:
                        self.final_box = []
                    self.t2_list.append(np.round((time.time() - t2) * 1000, 4))
                    # print(f'rnet time: {np.round(np.mean(self.t2_list), 4)} ms')
                    self.skip=0
                self.result = self.pbox
                self.last_gray = current_gray
                self.skip+=1
            else:
                self.result = self.cbox

def cof_test():
    device = torch.device('cpu')
    cof_obj = cof(device)
    video = mmcv.VideoReader('data/static_human/2.mp4')
    mean_ms = []

    # dat = dict(dir_name="queen", bboxes=[])
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('queen-cof.avi', fourcc, 60.0, (1280, 720))

    for frame in video:
        t = time.time()
        frame = cv2.resize(frame, (640, 480))
        cof_obj.detect(frame)
        # dat['bboxes'].append([cof_obj.result.x, cof_obj.result.y, cof_obj.result.w, cof_obj.result.h])
        mean_ms.append(np.round((time.time() - t) * 1000, 4))
        print(f'total time: {np.round(np.mean(mean_ms), 4)} ms')
        # print('-'*30)
        cv2.rectangle(frame,
                      (cof_obj.result.x, cof_obj.result.y),
                      (cof_obj.result.x+cof_obj.result.w, cof_obj.result.y+cof_obj.result.h),
                      (255,0,0), 2)
        cv2.putText(frame, str(np.round(np.mean(mean_ms), 4)) + "ms", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # out.write(frame)
        cv2.imshow("COF", frame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    # out.release()
    # file_out = 'results/queen/COF.json'
    # with open(file_out, 'w', encoding='utf-8') as fout:
    #     fout.write(json.dumps(dat))
    #     fout.write('\n')

def save_cof_result_video(root_path):
    device = torch.device('cuda')
    file_list = os.listdir(root_path)
    for file in file_list:
        file_out = f'results/{root_path.split("/")[1]}/COF_{file.split(".")[0]}.json'
        cof_obj = cof(device)
        video = mmcv.VideoReader(root_path+file)
        dat = dict(dir_name=root_path+file, bboxes=[])
        with open(file_out, 'w', encoding='utf-8') as fout:
            for img in video:
                cof_obj.detect(img)
                dat['bboxes'].append([cof_obj.result.x, cof_obj.result.y, cof_obj.result.w, cof_obj.result.h])
                # cv2.rectangle(img,
                #               (cof_obj.result.x, cof_obj.result.y),
                #               (cof_obj.result.x + cof_obj.result.w, cof_obj.result.y + cof_obj.result.h),
                #               (255, 0, 0), 2)
                # cv2.imshow(root_path+file, img)
                # k = cv2.waitKey(1)
                # if k == 27:
                #     cv2.destroyAllWindows()
                #     return
            print(len(dat['bboxes']))
            cv2.destroyAllWindows()
            fout.write(json.dumps(dat))
            fout.write('\n')
            del cof_obj

def save_mtcnn_result_video(root_path):
    device = torch.device('cuda')
    file_list = os.listdir(root_path)
    for file in file_list:
        file_out = f'results/{root_path.split("/")[1]}/MTCNN_{file.split(".")[0]}.json'
        video = mmcv.VideoReader(root_path + file)
        mtcnn = MTCNN(image_size=1080, thresholds=[0.9, 0.9, 0.7], min_face_size=40, device=device)
        dat = dict(dir_name=root_path + file, bboxes=[])
        with open(file_out, 'w', encoding='utf-8') as fout:
            for img in video:
                boxes, _ = mtcnn.detect(img)
                if boxes is None:
                    dat['bboxes'].append([0,0,0,0])
                else:
                    bb = boxes[0]
                    dat['bboxes'].append([int(bb[0]), int(bb[1]), int(bb[2]-bb[0]), int(bb[3]-bb[1])])
                    # cv2.rectangle(img,
                    #               (int(bb[0]), int(bb[1])),
                    #               (int(bb[2]), int(bb[3])),
                    #               (255, 0, 0), 2)
                    # cv2.imshow(file, img)
                    # k = cv2.waitKey(30)
                    # if k == 27:
                    #     cv2.destroyAllWindows()
                    #     return
            print(len(dat['bboxes']))
            cv2.destroyAllWindows()
            fout.write(json.dumps(dat))
            fout.write('\n')
        del mtcnn

if __name__=='__main__':
    cof_test()
    # test_mtchh_video()
