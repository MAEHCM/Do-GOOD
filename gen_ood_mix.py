import cv2
import random
import numpy as np


def generate_ood_data(output_path, strong_q_path, strong_a_path, weak_q_path, weak_a_path, weak_o_path,
                      data_num, visualize=True, img_w=800, img_h=1000, box_w_range=[150, 200], 
                      box_h_range=[20, 30], table_row_range=[8, 10], table_col_range=[2, 2], 
                      top_w_range=[500, 700], top_h_range=[50, 80], weak_padding=1, weak_prob=0.3):
    with open(strong_q_path, mode="r", encoding="utf-8") as f:
        strong_q_map = f.read().splitlines()
    with open(strong_a_path, mode="r", encoding="utf-8") as f:
        strong_a_map = f.read().splitlines()
    with open(weak_q_path, mode="r", encoding="utf-8") as f:
        weak_q_map = f.read().splitlines()
    with open(weak_a_path, mode="r", encoding="utf-8") as f:
        weak_a_map = f.read().splitlines()
    with open(weak_o_path, mode="r", encoding="utf-8") as f:
        weak_o_map = f.read().splitlines()

    output_lines = []
    visualize_output_lines = []

    prev, next = 0, 0
    # 遍历每张"图片"
    for e in range(data_num):
        # 先用other，question，answer填充表头
        top_w = random.randint(top_w_range[0], top_w_range[1])
        top_h = random.randint(top_h_range[0], top_h_range[1])
        print(f"top w: {top_w}, top h: {top_h}")
        top_x1, top_y1 = (img_w - top_w) // 2, 50
        index = random.randint(0, len(weak_o_map) - 1)
        entity_id = weak_o_map[index].split("\t")[-1]
        while index >= 0 and weak_o_map[index].split("\t")[-1] == entity_id:
            index -= 1
        index += 1
        output_line = "\t".join(weak_o_map[index].split("\t")[:-1])
        output_lines.append(output_line + f"\t{top_x1} {top_y1} {top_x1 + top_w} {top_y1 + top_h}")
        visualize_output_lines.append(output_line + f"\t{top_x1} {top_y1} {top_x1 + top_w} {top_y1 + top_h}\tH")
        index += 1
        for item in weak_o_map[index:]:
            if item.split("\t")[-1] != entity_id:
                break
            output_line = "\t".join(item.split("\t")[:-1])
            output_lines.append(output_line + f"\t{top_x1} {top_y1} {top_x1 + top_w} {top_y1 + top_h}")
            visualize_output_lines.append(output_line + f"\t{top_x1} {top_y1} {top_x1 + top_w} {top_y1 + top_h}\tH")

        # 随机确定question在最左或者最上
        top_q = True if random.randint(0, 1) == 1 else False
        # 强语义区域的行数和列数
        table_row = random.randint(table_row_range[0], table_row_range[1])
        table_col = random.randint(table_col_range[0], table_col_range[1])
        print(f"强语义区域行数：{table_row}，列数：{table_col}")
        # 总行数和列数
        total_row, total_col = table_row + weak_padding * 2, table_col + weak_padding * 2
        box_w, box_h = [], []
        total_w, total_h = 0, 0
        for _ in range(total_col):
            tmp = random.randint(box_w_range[0], box_w_range[1])
            box_w.append(tmp)
            total_w += tmp
        for _ in range(total_row):
            tmp = random.randint(box_h_range[0], box_h_range[1])
            box_h.append(tmp)
            total_h += tmp

        # 左上角box的x1和y1坐标，确保表格在图片中间
        init_x, init_y = (img_w - total_w) // 2, (img_h - total_h) // 2

        for r in range(total_row):
            for c in range(total_col):
                x1, y1 = init_x + sum(box_w[:c]), init_y + sum(box_h[:r])
                x2, y2 = x1 + box_w[c], y1 + box_h[r]

                # 强语义区域
                if weak_padding <= r and r < weak_padding + table_row and weak_padding <= c and c < weak_padding + table_col:
                    # Q在上和Q在下两种情况
                    if top_q and r == weak_padding:

                        index = random.randint(0, len(strong_q_map) - 1)
                        # 一直循环随机取token，直到遇到B开头的
                        while strong_q_map[index].split("\t")[-2] != "B-QUESTION":
                            index = random.randint(0, len(strong_q_map) - 1)
                        output_line = "\t".join(strong_q_map[index].split("\t")[:-1])
                        output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                        visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-QUESTION")
                        index += 1
                        for item in strong_q_map[index:]:
                            if item.split("\t")[-2] == "B-QUESTION":
                                break
                            output_line = "\t".join(item.split("\t")[:-1])
                            output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                            visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-QUESTION")

                    elif not top_q and c == weak_padding:

                        index = random.randint(0, len(strong_q_map) - 1)
                        # 一直循环随机取token，直到遇到B开头的
                        while strong_q_map[index].split("\t")[-2] != "B-QUESTION":
                            index = random.randint(0, len(strong_q_map) - 1)
                        output_line = "\t".join(strong_q_map[index].split("\t")[:-1])
                        output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                        visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-QUESTION")
                        index += 1
                        for item in strong_q_map[index:]:
                            if item.split("\t")[-2] == "B-QUESTION":
                                break
                            output_line = "\t".join(item.split("\t")[:-1])
                            output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                            visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-QUESTION")

                    else:
                        index = random.randint(0, len(strong_a_map) - 1)
                        # 一直循环随机取token，直到遇到B开头的
                        while strong_a_map[index].split("\t")[-2] != "B-ANSWER":
                            index = random.randint(0, len(strong_a_map) - 1)
                        output_line = "\t".join(strong_a_map[index].split("\t")[:-1])
                        output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                        visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-ANSWER")
                        index += 1
                        for item in strong_a_map[index:]:
                            if item.split("\t")[-2] == "B-ANSWER":
                                break
                            output_line = "\t".join(item.split("\t")[:-1])
                            output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                            visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tB-ANSWER")

                # 弱语义区域
                else:
                    if top_q:
                        if r == 0 or r == weak_padding:
                            # 取weak Q
                            if random.uniform(0, 1) < weak_prob:
                                index = random.randint(0, len(weak_q_map) - 1)
                                entity_id = weak_q_map[index].split("\t")[-1]
                                while index >= 0 and weak_q_map[index].split("\t")[-1] == entity_id:
                                    index -= 1
                                index += 1
                                output_line = "\t".join(weak_q_map[index].split("\t")[:-1])
                                output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tQ")
                                index += 1
                                for item in weak_q_map[index:]:
                                    if item.split("\t")[-1] != entity_id:
                                        break
                                    output_line = "\t".join(item.split("\t")[:-1])
                                    output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                    visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tQ")
                        elif weak_padding + table_row == r or weak_padding + table_row - 1 == r:
                            # 取weak A
                            if random.uniform(0, 1) < weak_prob:
                                index = random.randint(0, len(weak_a_map) - 1)
                                entity_id = weak_a_map[index].split("\t")[-1]
                                while index >= 0 and weak_a_map[index].split("\t")[-1] == entity_id:
                                    index -= 1
                                index += 1
                                output_line = "\t".join(weak_a_map[index].split("\t")[:-1])
                                output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tA")
                                index += 1
                                for item in weak_a_map[index:]:
                                    if item.split("\t")[-1] != entity_id:
                                        break
                                    output_line = "\t".join(item.split("\t")[:-1])
                                    output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                    visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tA")
                    elif not top_q:
                        if c == 0 or c == weak_padding:
                            # 取weak Q
                            if random.uniform(0, 1) < weak_prob:
                                index = random.randint(0, len(weak_q_map) - 1)
                                entity_id = weak_q_map[index].split("\t")[-1]
                                while index >= 0 and weak_q_map[index].split("\t")[-1] == entity_id:
                                    index -= 1
                                index += 1
                                output_line = "\t".join(weak_q_map[index].split("\t")[:-1])
                                output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tQ")
                                index += 1
                                for item in weak_q_map[index:]:
                                    if item.split("\t")[-1] != entity_id:
                                        break
                                    output_line = "\t".join(item.split("\t")[:-1])
                                    output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                    visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tQ")
                        elif weak_padding + table_col == c or weak_padding + table_col - 1 == c:
                            # 取weak A
                            if random.uniform(0, 1) < weak_prob:
                                index = random.randint(0, len(weak_a_map) - 1)
                                entity_id = weak_a_map[index].split("\t")[-1]
                                while index >= 0 and weak_a_map[index].split("\t")[-1] == entity_id:
                                    index -= 1
                                index += 1
                                output_line = "\t".join(weak_a_map[index].split("\t")[:-1])
                                output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tA")
                                index += 1
                                for item in weak_a_map[index:]:
                                    if item.split("\t")[-1] != entity_id:
                                        break
                                    output_line = "\t".join(item.split("\t")[:-1])
                                    output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}")
                                    visualize_output_lines.append(output_line + f"\t{x1} {y1} {x2} {y2}\tA")

        # 每张“图片”用空行隔开
        output_lines.append("")
        visualize_output_lines.append("")
        
        # 仅用于可视化
        if visualize:
            prev = next
            next = len(visualize_output_lines)

            img_path = f"/{e}.png"
            img = np.ones((img_h, img_w, 3), dtype=np.uint8)
            img *= 255
            for output_line in visualize_output_lines[prev:next]:
                if output_line == "":
                    continue

                box = output_line.split("\t")[-2]
                pt1 = (int(box.split(" ")[0]), int(box.split(" ")[1]))
                pt2 = (int(box.split(" ")[2]), int(box.split(" ")[3]))
                label = output_line.split("\t")[-1]
                if label == "Q":
                    cv2.rectangle(img, pt1, pt2, (0, 100, 0), 4)
                elif label == "A":
                    cv2.rectangle(img, pt1, pt2, (0, 0, 100), 4)
                elif label == "H":
                    cv2.rectangle(img, pt1, pt2, (0, 0, 0), 4)
                elif label == "B-QUESTION":
                    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)
                elif label == "B-ANSWER":
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 4)

            cv2.imwrite(img_path, img)

    with open(output_path, mode="w", encoding="utf-8") as f:
        for output_line in output_lines:
            f.write(output_line + "\n")

if __name__ == "__main__":
    generate_ood_data("mix_test.txt", "/strong_question_map",
                      "/strong_answer_map", "/weak_Q_map",
                      "/weak_A_map", "/weak_other_map",50)
