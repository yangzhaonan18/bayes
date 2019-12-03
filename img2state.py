import cv2

class State():
    def __init__(self, T_path, V_path, E_path):
        W_num = int(16.5 * 2)
        H_num = int(38.5 * 2)
        W_one = 1000 / W_num
        H_one = W_one  * 1.74 / 2
        W = 1000
        H = 500
        T_img = cv2.imread(T_path, cv2.IMREAD_COLOR)
        V_img = cv2.imread(V_path, cv2.IMREAD_COLOR)
        self.T_img = cv2.resize(T_img, (W, H))
        self.V_img = cv2.resize(V_img, (W, H))
        print(self.T_img.shape)  # (518, 1033, 3)
        print(self.V_img.shape)  # (518, 1033, 3)

        for i in range(1, H_num, 1):
            if i % 2 == 1:
                for j in range(1, W_num, 1):
                    if j % 3 == 1 :
                        print("i, j =", i, j)
                        cv2.circle(self.T_img, center=(int(i * W_one), int(j * H_one)), radius=2, color=(255, 0, 0), thickness=2)
        print(i *  j / 6)
        cv2.imshow("a", self.T_img)
        cv2.waitKey(0)



        


    def T():
        pass

    def V():
        pass

    def E():
        pass












# cv2.imshow("a", T_img)
# cv2.waitKey(10)

if __name__ =="__main__":
    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.img"

    state = State(T_path, V_path, E_path)
    # T = state.T()




