# -*-coding:utf-8-*-
import json, os, math
import matplotlib.pyplot as plt
import numpy as np

def stability_compute():
    num = 0
    for file in file_set:
        with open(root_path + file, 'r', encoding='utf-8') as outf:
            lines = outf.readlines()
        data = json.loads(lines[0].strip('\r\n'))
        bboxes = data['bboxes']
        stability = []
        smoothness = []
        last_box = [0, 0, 0, 0]
        for cur_box in bboxes:
            if 0 in cur_box:
                last_box = [0, 0, 0, 0]
                continue
            if sum(last_box)==0:
                last_box = cur_box
            else:
                pt1_offset = ((cur_box[0] - last_box[0]) ** 2 + (cur_box[1] - last_box[1]) ** 2) ** 0.5
                pt2_offset = ((cur_box[0] + cur_box[2] - last_box[0] - last_box[2]) ** 2 + (cur_box[1] - last_box[1]) ** 2) ** 0.5
                pt3_offset = ((cur_box[0] + cur_box[2] - last_box[0] - last_box[2]) ** 2 + (cur_box[1] + cur_box[3] - last_box[1] - last_box[3]) ** 2) ** 0.5
                pt4_offset = ((cur_box[0]- last_box[0]) ** 2 + (cur_box[1] + cur_box[3] - last_box[1] - last_box[3]) ** 2) ** 0.5
                c_offset = ((cur_box[0] + cur_box[2] * 0.5 - last_box[0] - last_box[2] * 0.5) ** 2 + (cur_box[1] + cur_box[3] * 0.5 - last_box[1] - last_box[3] * 0.5) ** 2) ** 0.5
                w_offset = abs(cur_box[2]-last_box[2])
                h_offset = abs(cur_box[3]-last_box[3])
                # sta += (pt1_offset + pt2_offset + pt3_offset + pt4_offset + c_offset) / (5*num)
                smoothness.append(math.log((math.exp((pt1_offset + pt2_offset + pt3_offset + pt4_offset)/4))/math.exp(c_offset)))
                stability.append((w_offset + h_offset + c_offset)/3)
                last_box = cur_box
        # print(f"{file} smoothness: {smoothness} stability: {stability}")
        print(f"{np.round(np.mean(stability), 6)} & ", end="")
        num+=1
        if num in [5, 10, 15]:
            print("")
            
def point_plot(one_graph):
    p_set = ['top-left', 'top-right', 'bottom-right', 'bottom-left', 'center']
    p = p_set[4]
    if one_graph:
        fig = plt.figure(figsize=(15, 20))
    for idx, file in enumerate(file_set):
        with open(root_path + file, 'r', encoding='utf-8') as outf:
            lines = outf.readlines()
        data = json.loads(lines[0].strip('\r\n'))
        bboxes = data['bboxes']
        cx = []
        cy = []
        for cur_box in bboxes:
            if 0 in cur_box:
                continue
            if p==p_set[0]:
                cx.append(max(0, cur_box[0]))
                cy.append(cur_box[1])
            elif p==p_set[1]:
                cx.append(cur_box[0] + cur_box[2])
                cy.append(cur_box[1])
            elif p==p_set[2]:
                cx.append(cur_box[0] + cur_box[2])
                cy.append(cur_box[1] + cur_box[3])
            elif p==p_set[3]:
                cx.append(cur_box[0])
                cy.append(cur_box[1] + cur_box[3])
            elif p==p_set[4]:
                cx.append(max(0, cur_box[0]) + cur_box[2] * 0.5)
                cy.append(max(0, cur_box[1]) + cur_box[3] * 0.5)
        cx = [(x-min(cx))/(max(cx)-min(cx)) for x in cx]
        cy = [(y-min(cy))/(max(cy)-min(cy)) for y in cy]
        if one_graph:
            idx_list = [1,4,7,10,13, 2,5,8,11,14, 3,6,9,12,15]
            ax = fig.add_subplot(5, 3, idx_list[idx])
            ax.plot(cx, cy)
            if idx == 9:
                plt.xlabel('x', fontsize=20)
            if idx < 5:
                plt.ylabel(f'sub-video {file.split(".")[0].split("_")[1]}', fontsize=15, rotation = 0,  ha='right')
            if idx == 0:
                plt.title('C-OF', fontsize=20)
            if idx == 5:
                plt.title('MDNet', fontsize=20)
            if idx == 10:
                plt.title('MTCNN', fontsize=20)
            if idx_list[idx]>12:
                plt.xlim(0, 1)
            else:
                plt.xticks([])
            plt.yticks([])
            if idx >9:
                ax2 = ax.twinx()
            if idx == 12:
                ax2.set_ylabel("y", fontsize=20, rotation=0, ha='left')


            # ax = fig.add_subplot(3, 5, idx+1)
            # ax.plot(cx, cy)
            # if idx==12:
            #     plt.xlabel('x', fontsize=20)
            # if idx==0:
            #     plt.ylabel('C-OF', fontsize=20, rotation = 0,  ha='right')
            # if idx==5:
            #     plt.ylabel('MDNet', fontsize=20, rotation=0,  ha='right')
            # if idx==10:
            #     plt.ylabel('MTCNN', fontsize=20, rotation=0,  ha='right')
            # if idx>9:
            #     plt.xlim(0, 1)
            # else:
            #     plt.xticks([])
            # plt.yticks([])
            # if idx ==4:
            #     ax2 = ax.twinx()
            # if idx ==9:
            #     ax2 = ax.twinx()
            #     ax2.set_ylabel("y", fontsize=20, rotation = 0,  ha='left')
            # if idx ==14:
            #     ax2 = ax.twinx()
            # if idx < 5:
            #     plt.title(f'sub-video {file.split(".")[0].split("_")[1]}', fontsize=15)
        else:
            plt.plot(cx, cy)
            plt.xlabel('x', fontsize=fontsize)
            plt.ylabel('y', rotation = 0,  ha='right', fontsize=fontsize)
            plt.suptitle(f'{file.split(".")[0].split("_")[0]}, {v.replace("_", " ")} sub-video {file.split(".")[0].split("_")[1]}', fontsize=fontsize)
            plt.savefig(f'{root_path.replace("box_info", "point_route")}/sh_{file.split(".")[0]}_{p}.png')
            plt.show()

    if one_graph:
        plt.suptitle(f'Normalized {p} point track on {v.replace("_", " ")} videos', fontsize=25, y=0.94)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f'{root.replace("box_info", "point_route")}/{v}_{p}.png')
        plt.show()


def get_gt(root_path):
    with open(root_path + "GT.json", 'r', encoding='utf-8') as outf:
        lines = outf.readlines()
    data = json.loads(lines[0].strip('\r\n'))
    gt_bboxes = data['bboxes']
    return gt_bboxes


if __name__=='__main__':
    v = "active_camera"
    root = '/home/hans/WorkSpace/COF/results/box_info/'
    root_path = root+f'/{v}/'
    file_set = os.listdir(root_path)
    file_set.sort()
    fontsize=20
    # file_set.remove("GT.json")
    # stability_compute()    # success_plot()
153

    point_plot(one_graph=True)
