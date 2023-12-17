import tkinter as tk
from tkinter import messagebox

# 姓氏预测函数
def predict_lastname(name, include_double_surname):
    # 在这里编写姓氏预测的逻辑
    lastname_dict=False
    if include_double_surname:
        # 添加一些复姓
        lastname_dict = True
    if lastname_dict:
        return name[:2]
    else:
        return name[0]

# 处理按钮点击事件
def on_predict():
    name = name_entry.get()
    include_double_surname = double_surname_var.get()
    if name:
        lastname = predict_lastname(name, include_double_surname)
        messagebox.showinfo("预测结果", f"预测的姓氏为：{lastname}")
    else:
        messagebox.showwarning("警告", "请输入名字！")

# 创建GUI窗口
window = tk.Tk()
window.title("姓氏预测")
window.geometry("300x250")

# 创建标签和文本框
name_label = tk.Label(window, text="请输入名字：")
name_label.pack()
name_entry = tk.Entry(window)
name_entry.pack()

# 创建复选框
double_surname_var = tk.BooleanVar()
double_surname_checkbox = tk.Checkbutton(
    window,
    text="包括复姓",
    variable=double_surname_var
)
double_surname_checkbox.pack()

# 创建预测按钮
predict_button = tk.Button(window, text="预测", command=on_predict)
predict_button.pack()

# 运行窗口
window.mainloop()
