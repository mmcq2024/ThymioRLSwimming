import subprocess
import sys

# Automatically install required packages if missing
required = {
    "tdmclient": "tdmclient",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pillow": "PIL"  # Actual import name is PIL
}

for pip_name, import_name in required.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for package in required:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Special handling for tkinter (not installable via pip)
try:
    import tkinter
except ImportError:
    print("Error: tkinter is not installed. On Linux, try:\n  sudo apt-get install python3-tk")
    sys.exit(1)

import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import SwimmingThymio_Segmented as sts
from tkinter import PhotoImage
import os
from PIL import Image, ImageTk


previous_frame = None  # Global tracker
last_frame_level1 = None
last_frame_level2 = None
q_table_update_callback = None  # This will be set later by the GUI
current_level = 1  # Default to Level 1

frame_stack_level1 = []
frame_stack_level2 = []

has_initialized_thymio = False
training_step_running = False
reward_button = None

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left",
                         background="#ffffe0", relief="solid", borderwidth=1,
                         font=("Segoe UI emoji", 10))
        label.pack(ipadx=5, ipady=2)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# --- Async and Navigation Helpers ---
def run_async(coro):
    threading.Thread(target=lambda: asyncio.run(coro())).start()

def go_back():
    if last_level == 1 and len(frame_stack_level1) > 1:
        frame_stack_level1.pop()  # Remove current
        switch_frame(frame_stack_level1.pop())  # Show previous
    elif last_level == 2 and len(frame_stack_level2) > 1:
        frame_stack_level2.pop()
        switch_frame(frame_stack_level2.pop())

def run_test_then_next():
    async def wrapper():
        await sts.run_test(test_var.get())
        after_test()
    run_async(wrapper)

def run_final_test_then_next():
    async def wrapper():
        await sts.run_test(test_var.get())
        if current_level == 2:
            switch_frame(frame13)
        else:
            after_final_test()
    run_async(wrapper)

def run_train_then_next():
    async def wrapper():
        await sts.run_train(train_var.get(), level=current_level)
        after_training()
    run_async(wrapper)

def run_train_and_continue():
    global has_initialized_thymio

    if not has_initialized_thymio:
        # Show popup error
        error_popup = tk.Toplevel(root)
        error_popup.title("Error")
        error_popup.configure(bg="#f9f9f9")
        tk.Label(error_popup, text="⚠️ Please run GET STARTED before training and try again! It takes 30 seconds to initialize. You should see the Thymio move.", font=("Segoe UI", 12, "bold"),
                 bg="#f9f9f9", wraplength=300).pack(padx=20, pady=20)
        tk.Button(error_popup, text="Go to GET STARTED", command=lambda: [switch_frame(frame1), error_popup.destroy()],
                  bg="#388994", fg="white", font=("Segoe UI", 11, "bold"), padx=10, pady=5).pack(pady=10)
        return

    async def wrapper():
        await sts.run_train(train_var.get(), switch_to_final=True, level=current_level)
        if current_level == 2:
            switch_frame(frame12)
        else:
            after_training()
    run_async(wrapper)

def switch_frame(new_frame):
    global last_activity_frame, last_level, last_frame_level1, last_frame_level2

    # Push to corresponding stack
    if new_frame in [frame1, frame2, frame3, frame4, frame5, frame6]:
        last_activity_frame = new_frame
        last_level = 1
        last_frame_level1 = new_frame
        if not frame_stack_level1 or frame_stack_level1[-1] != new_frame:
            frame_stack_level1.append(new_frame)
    elif new_frame in [frame7, frame8, frame9, frame10, frame11, frame12, frame13]:
        last_activity_frame = new_frame
        last_level = 2
        last_frame_level2 = new_frame
        if not frame_stack_level2 or frame_stack_level2[-1] != new_frame:
            frame_stack_level2.append(new_frame)

    # Hide all, show new
    for f in frames:
        f.pack_forget()
    new_frame.pack(fill="both", expand=True)


def next_from_intro():
    switch_frame(mode_select_frame)

def start_activity1():
    global current_level
    current_level = 1
    sts.restart_learning()
    switch_frame(frame1)

def start_activity2():
    global current_level
    current_level = 2
    sts.restart_learning()
    switch_frame(frame7)

def after_get_started():
    global has_initialized_thymio
    has_initialized_thymio = True
    print("[DEBUG] Thymio is now initialized")

    if current_level == 2:
        switch_frame(frame10)
    else:
        switch_frame(frame2)



def after_test():
    switch_frame(frame3)

def after_training():
    switch_frame(frame4)

def after_final_test():
    switch_frame(frame5)

def handle_restart_learning():
    sts.restart_learning()
    if current_level == 1:
        switch_frame(frame3)  # Go back to training step
    else:
        switch_frame(frame10)

def open_menu():
    # Create a floating popup like a dropdown
    menu_popup = tk.Toplevel(root)
    menu_popup.overrideredirect(True)  # Removes window borders
    menu_popup.configure(bg="#e6e6e6")
    menu_popup.geometry("+10+50")  # Position near ☰

    def close_menu(event=None):
        menu_popup.destroy()

    menu_popup.bind("<FocusOut>", close_menu)
    menu_popup.bind("<Escape>", close_menu)
    menu_popup.focus_force()

    # Title
    tk.Label(menu_popup, text="MENU", font=("Gill Sans MT", 16, "bold"),
             bg="#e6e6e6", fg="#003366").pack(pady=10)

    # Return to home screen
    tk.Button(menu_popup, text="🏠 Return to Home",
              command=lambda: [switch_frame(mode_select_frame), close_menu()],
              bg="#388994", fg="white", font=("Microsoft YaHei", 12, "bold"),
              bd=0, padx=10, pady=5, width=30).pack(pady=5)

    # Go to Activity 1
    def go_to_level1():
        sts.restart_learning()
        switch_frame(frame1)
        close_menu()

    # Go to Activity 2
    def go_to_level2():
        sts.restart_learning()
        switch_frame(frame7)
        close_menu()

    # Trigger online training
    def go_to_online_training():
        switch_frame(online_training_frame)
        close_menu()

    # Level 1
    tk.Button(menu_popup, text="Go to Level 1 Activity",
              command=go_to_level1,
              bg="#388994", fg="white", font=("Microsoft YaHei", 12, "bold"),
              bd=0, padx=10, pady=5, width=30).pack(pady=5)

    # Level 2
    tk.Button(menu_popup, text="Go to Level 2 Activity",
              command=go_to_level2,
              bg="#388994", fg="white", font=("Microsoft YaHei", 12, "bold"),
              bd=0, padx=10, pady=5, width=30).pack(pady=5)

    # Online Training
    tk.Button(menu_popup, text="Virtual Training",
              command=go_to_online_training,
              bg="#388994", fg="white", font=("Microsoft YaHei", 12, "bold"),
              bd=0, padx=10, pady=5, width=30).pack(pady=5)
    
def go_to_online_training():
    switch_frame(online_training_frame)

def render_q_table(frame, show_test_button=False):
    # Clear the current frame
    for widget in frame.winfo_children():
        widget.destroy()

    q = sts.get_q_table()
    states = list(q.keys())
    actions = list(q[states[0]].keys())

    tk.Label(frame, text="📊 Current Q-Table", **header_style).pack(pady=10)

    state_labels = {
        "R0, L0": "Right Arm Front\nLeft Arm Front",
        "R0, L180": "Right Arm Front\nLeft Arm Back",
        "R180, L0": "Right Arm Back\nLeft Arm Front",
        "R180, L180": "Right Arm Back\nLeft Arm Back"
    }

    action_labels = {
        0: "Move Right arm",
        1: "Move Left arm",
        2: "Move both arms"
    }

    # Show live state-action info only during training
    info = sts.get_live_info()

    info_frame = tk.Frame(frame, bg="#67d7e6")
    info_frame.pack(pady=15)

    # Convert raw values to descriptions
    initial_state_desc = state_labels.get(info['state'], info['state'])
    action_number = int(info['action'][1]) if info['action'].startswith("A") else info['action']
    action_desc = action_labels.get(action_number, info['action'])

    row_frame = tk.Frame(info_frame, bg="#67d7e6")
    row_frame.pack(anchor="w")
    tk.Label(row_frame, text="Initial State: ", font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").pack(side="left")
    tk.Label(row_frame, text=initial_state_desc, font=("Microsoft YaHei", 12), bg="#67d7e6").pack(side="left")

    row_frame = tk.Frame(info_frame, bg="#67d7e6")
    row_frame.pack(anchor="w")
    tk.Label(row_frame, text="Action: ", font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").pack(side="left")
    tk.Label(row_frame, text=action_desc, font=("Microsoft YaHei", 12), bg="#67d7e6").pack(side="left")

    row_frame = tk.Frame(info_frame, bg="#67d7e6")
    row_frame.pack(anchor="w")
    tk.Label(row_frame, text="Reward: ", font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").pack(side="left")
    tk.Label(row_frame, text=info['reward'], font=("Microsoft YaHei", 12), bg="#67d7e6").pack(side="left")

    row_frame = tk.Frame(info_frame, bg="#67d7e6")
    row_frame.pack(anchor="w")
    tk.Label(row_frame, text="Future State: ", font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").pack(side="left")
    tk.Label(row_frame, text=state_labels.get(info['next_state'], info['next_state']), font=("Microsoft YaHei", 12), bg="#67d7e6").pack(side="left")

    table = tk.Frame(frame, bg="#67d7e6")
    table.pack()

    # Column headers
    tk.Label(table, text="State / Action", font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").grid(row=0, column=0)

    for j, action in enumerate(actions):
        tk.Label(table, text=action_labels[action], font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6").grid(row=0, column=j+1)

    # Fill in the Q-values
    for i, state in enumerate(states):
        tk.Label(table, text=state_labels[state], font=("Microsoft YaHei", 12, "bold"), bg="#67d7e6", justify="left").grid(row=i+1, column=0)
        for j, action in enumerate(actions):
            value = q[state][action]
            tk.Label(table, text=f"{value:.2f}", font=("Microsoft YaHei", 12), bg="#67d7e6").grid(row=i+1, column=j+1)

    if not show_test_button and current_level == 2:
        # Detect final step
        if hasattr(sts.run_train_step_level2, "episode") and sts.run_train_step_level2.episode >= train_var.get():
            # Final step reached → show "Finalize and Test"
            tk.Button(
                frame,
                text="FINALIZE AND TEST",
                command=lambda: switch_frame(frame12),
                **button_style
            ).pack(pady=20)
        else:
            # Normal case → continue training
            tk.Button(
                frame,
                text="NEXT ACTION",
                command=run_level2_step,
                **button_style
            ).pack(pady=20)
        
def gui_callback(is_final=False):
    if last_level == 2:  # Only show Q-table during Level 2
        render_q_table(frame11.content, show_test_button=is_final)
        switch_frame(frame11)

sts.set_gui_callback(gui_callback)

def show_qtable_frame():
    render_q_table(frame11.content)
    switch_frame(frame11)

def show_manual_popup():
    popup = tk.Toplevel(root)
    popup.title("Thymio Manual Control")
    popup.configure(bg="#e6e6e6")
    popup.geometry("600x400")

    tk.Label(popup, text="THYMIO MANUAL CONTROL", font=("Gill Sans MT", 20, "bold"),
             fg="#003366", bg="#e6e6e6", wraplength=600, justify="center").pack(pady=10)
    tk.Label(popup, text="Modify the parameters depending on how many iterations you want to complete. If you are in the middle of a training or a testing phase, the change will not be reflected.",  font=("Microsoft YaHei Light", 14), bg="#e6e6e6", wraplength=500, justify="center").pack(pady=5)

    control_frame = tk.Frame(popup, bg="#e6e6e6")
    control_frame.pack(pady=10)

    tk.Label(control_frame, text="Parameters", font=("Microsoft YaHei Light", 12, "bold"),
         bg="#e6e6e6").grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="n")

    # 🟦 Row 1: Actions to be taken during testing
    tk.Label(control_frame, text="Actions to be taken during testing:", font=("Microsoft YaHei Light", 12, "italic"),
             bg="#e6e6e6").grid(row=1, column=0, padx=10, pady=5)
    tk.Spinbox(control_frame, from_=1, to=50, width=5, textvariable=test_var).grid(row=1, column=1, padx=10, pady=5)

    # 🟦 Row 2: Repetitions for physical training
    tk.Label(control_frame, text="Repetitions for physical training:", font=("Microsoft YaHei Light", 12, "italic"),
             bg="#e6e6e6").grid(row=2, column=0, padx=10, pady=5)
    tk.Spinbox(control_frame, from_=1, to=100, width=5, textvariable=train_var).grid(row=2, column=1, padx=10, pady=5)

    # 🟦 Row 3: Virtual Training Iterations
    tk.Label(control_frame, text="Virtual Training Iterations:", font=("Microsoft YaHei Light", 12, "italic"),
             bg="#e6e6e6").grid(row=3, column=0, padx=10, pady=5)
    tk.Spinbox(control_frame, from_=1, to=100, width=5, textvariable=pretrain_var).grid(row=3, column=1, padx=10, pady=5)

    
    tk.Button(popup, text="Close", command=popup.destroy, bg="#666666", fg="white",
          font=("Microsoft YaHei", 12, "bold"), padx=10, pady=6, bd=0, width=42).pack(pady=10)

def show_virtual_training_demo():
    async def demo():
        await sts.run_pretrain(pretrain_var.get())  # Use teacher-set iterations
        await sts.run_test(test_var.get())          # Show what was learned
    run_async(demo)

def show_testing_image_popup():
    popup = tk.Toplevel(root)
    popup.title("Correct vs Incorrect Swimming Behavior")
    popup.configure(bg="#e6e6e6")
    popup.geometry("500x500+20+60")  # Smaller and top-left corner
    img_label = tk.Label(popup, image=correct_photo, bg="#e6e6e6")
    img_label.pack(padx=10, pady=10)
    popup.correct_photo = correct_photo  # Keep reference

def run_level2_step():
    """Execute an action, update the Q-table using expected reward, and show it immediately (no buttons)."""
    global has_initialized_thymio, training_step_running
    if not has_initialized_thymio:
        print("[DEBUG] Training attempted before GET STARTED")

        # 2. Show an error popup to explain it
        error_popup = tk.Toplevel(root)
        error_popup.title("Thymio Not Initialized")
        error_popup.configure(bg="#f9f9f9")

        tk.Label(
            error_popup,
            text="⚠️ Please run GET STARTED before training and try again! It takes 30 seconds to initialize. You should see the Thymio move.",
            font=("Segoe UI", 12, "bold"),
            bg="#f9f9f9",
            wraplength=300
        ).pack(padx=20, pady=20)

        # 3. Add a button to immediately run GET STARTED
        tk.Button(
            error_popup,
            text="Run GET STARTED",
            command=lambda: [
                run_async(lambda: asyncio.run(sts.run_get_started()) or after_get_started()),
                error_popup.destroy()
            ],
            bg="#388994",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=5
        ).pack(pady=10)

        return  # 4. Don't continue with training

    # ✅ If we got here, Thymio is initialized and we can run the action
    training_step_running = True

    async def step():
        await sts.run_train_step_level2()
        await sts.confirm_reward_and_update_level2(level=2)
        training_step_running = False

    threading.Thread(target=lambda: asyncio.run(step())).start()


def reward_and_continue_level2():
    """After student gives reward, update Q-table."""
    global training_step_running, reward_button
    if reward_button:
        reward_button.pack_forget()

    async def step():
        await sts.confirm_reward_and_update_level2()
        training_step_running = False

    threading.Thread(target=lambda: asyncio.run(step())).start()

def run_virtual_training_with_popup():
    # 🚫 Temporarily disable GUI updates
    original_callback = sts.q_table_gui_callback
    sts.set_gui_callback(lambda *args, **kwargs: None)

    # Run the virtual training
    sts.run_pretrain(pretrain_var.get())

    # ✅ Restore callback
    sts.set_gui_callback(original_callback)

    # Show a confirmation popup only
    popup = tk.Toplevel(root)
    popup.title("Virtual Training Done")
    popup.configure(bg="#f9f9f9")

    tk.Label(
        popup,
        text="✅ Virtual training completed!\nYou may now test Thymio’s learned behavior.",
        font=("Segoe UI", 12),
        bg="#f9f9f9",
        wraplength=300
    ).pack(padx=20, pady=20)

    tk.Button(
        popup,
        text="OK",
        command=popup.destroy,
        bg="#388994",
        fg="white",
        font=("Segoe UI", 11, "bold"),
        padx=10,
        pady=5
    ).pack(pady=10)

def start_testing():
    show_testing_image_popup()
    sts.start_testing_loop_level1(100)

def stop_testing():
    sts.stop_testing_loop_level1()
    if last_activity_frame == frame12:
        switch_frame(frame13)  # Move directly to reflection after Level 2 testing

def set_initialized():
    global has_initialized_thymio
    has_initialized_thymio = True

# --- GUI setup ---
root = tk.Tk()
root.title("🏊 Teach Thymio To Swim")
root.geometry("1000x700")
root.configure(bg="#67d7e6")

pretrain_var = tk.IntVar(value=10)
train_var = tk.IntVar(value=15)
test_var = tk.IntVar(value=10)

frames = [tk.Frame(root, bg="#67d7e6") for _ in range(19)]
intro_frame, mode_select_frame, nav_frame, menu_frame, frame1, frame2, frame3, frame4, frame5, frame6, \
frame7, frame8, frame9, frame10, frame11, frame12, frame13, online_training_frame, virtual_done_frame = frames

# --- Load Images Once ---
images_dir = os.path.join(os.path.dirname(__file__), "Images")

wave_path = os.path.join(images_dir, "waves.PNG")
pil_wave = Image.open(wave_path)
wave_img = ImageTk.PhotoImage(pil_wave)

from itertools import count
gif_path = os.path.join(images_dir, "SwimmingThymio.gif")
gif = tk.PhotoImage(file=gif_path)
# Get all frames from GIF
intro_gif_frames = [tk.PhotoImage(file=gif_path, format=f'gif -index {i}') for i in range(2)]   

correct_path = os.path.join(images_dir, "correct_incorrect.png")
correct_img = Image.open(correct_path)
correct_img = correct_img.resize((400, 400))  # Resize as needed
correct_photo = ImageTk.PhotoImage(correct_img)

# First define all frame structures
for frame in frames:
    content_frame = tk.Frame(frame, bg="#67d7e6")
    content_frame.pack(expand=True, fill="both")
    frame.content = content_frame

    inner_center = tk.Frame(content_frame, bg="#67d7e6")
    inner_center.place(relx=0.5, rely=0.5, anchor="center")
    frame.inner_center = inner_center

    center_block = tk.Frame(inner_center, bg="#67d7e6")
    center_block.pack()
    frame.center_block = center_block

# Then add the wave background only for frame6 and frame7
for wf in [frame6]:
    wave_label = tk.Label(wf.content, image=wave_img, bg="#67d7e6", bd=0, highlightthickness=0)
    wave_label.image = wave_img
    wave_label.place(relx=0.5, rely=1.0, anchor="s", relwidth=1.0)
    wave_label.lower()

for frame in [frame2, frame3, frame4, frame5, frame6,
              frame8, frame9, frame10, frame11, frame12, frame13]:
    tk.Button(
        frame.content,
        text="←",
        font=("Segoe UI", 16, "bold"),
        bg="#67d7e6",         
        fg="#003366",
        bd=0,
        activebackground="#e0e0e0",
        command=go_back
    ).place(x=40, y=0)


button_style = {
    "bg": "#388994",
    "fg": "white",
    "font": ("Microsoft YaHei Light", 16, "bold"),
    "padx": 12,
    "pady": 10,
    "bd": 0,
    "width": 20
}

menu_style = {
    "bg": "#67d7e6",         # same as background for transparent effect
    "fg": "#003366",         # dark text for visibility
    "font": ("Segoe UI", 14, "bold"),
    "bd": 0,
    "activebackground": "#67d7e6",  # prevent color change on hover
    "highlightthickness": 0,
    "relief": "flat"
}

label_style = {
    "bg": "#67d7e6",
    "font": ("Microsoft YaHei Light", 14),
    "wraplength": 800,
    "justify": "center"
}

header_style = {
    "bg": "#67d7e6",
    "font": ("Gill Sans MT", 20, "bold"),
    "fg": "#003366",
    "wraplength": 600,
    "justify": "center"
}

# Add info icon in training and testing frames
for f in [frame2, frame3, frame4]:
    info_btn = tk.Button(f.content, text="Correct behavior ❓", command=show_testing_image_popup, **menu_style)
    info_btn.place(x=800, y=5)


# --- Intro Frame ---
tk.Label(intro_frame.center_block, text="WELCOME TO THE \n THYMIO SWIMMING ACTIVITY!", **header_style).pack(pady=(0, 10))

tk.Label(intro_frame.center_block, text="An interactive experience where you can teach students about\nReinforcement Learning.",
         font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack(pady=(0, 20))

# Animated GIF label
intro_img_label = tk.Label(intro_frame.center_block, bg="#67d7e6")
intro_img_label.pack(pady=(0, 20))

# Animation function
def animate_intro_gif(frame_index=0):
    frame = intro_gif_frames[frame_index]
    intro_img_label.configure(image=frame)
    intro_img_label.image = frame
    root.after(500, animate_intro_gif, (frame_index + 1) % len(intro_gif_frames))

animate_intro_gif()

# Final text and button
tk.Label(intro_frame.center_block, text="Click the arrow below to get started!", **label_style).pack(pady=(0, 10))
tk.Button(intro_frame.center_block, text="→", font=("Helvetica", 40, "bold"), width=3, height=1, bg="#67d7e6", fg="black", bd=0,
          command=next_from_intro).pack(pady=(0, 10))

# --- Mode Selection Frame ---
tk.Label(mode_select_frame.center_block, text="HOME", **header_style).pack(pady=(10, 5))

tk.Label(mode_select_frame.center_block,
         text="Choose the option you want to complete.",
         font=("Microsoft YaHei Light", 14, "italic"),
         bg="#67d7e6").pack(pady=(0, 10))

tk.Label(mode_select_frame.center_block, text="Level 1 is meant to introduce the concept of giving rewards to Thymio to teach a front crawl. No prerequisites required for students.", **label_style).pack(pady=(0, 10))
tk.Button(mode_select_frame.center_block, text="START LEVEL 1 ACTIVITY", command=start_activity1, **button_style).pack(pady=5)

tk.Label(mode_select_frame.center_block, text="Level 2 is meant to introduce the inner workings of a Q-learning algorithm, focusing on the creation of a Q-table. Students must understand arithmetic operations, order of operations and reading tables.", **label_style).pack(pady=(10, 10))
tk.Button(mode_select_frame.center_block, text="START LEVEL 2 ACTIVITY", command=start_activity2, **button_style).pack(pady=5)

tk.Label(mode_select_frame.center_block, text="Virtual Training Preview allows you to see what ideal behavior looks like without student input. It does the training demonstrated in level 1 automatically so Thymio is able to swim.", **label_style).pack(pady=(20, 10))
tk.Button(mode_select_frame.center_block, text="VIRTUAL TRAINING", command=lambda: switch_frame(online_training_frame), **button_style).pack(pady=5)


# --- Navigation Frame (Dropdown style) ---
tk.Label(nav_frame.content, text="MENU", **header_style).pack(pady=20)

# Return to Level 1 (or start at Step 1 if not visited)
tk.Button(nav_frame.content, text="Go to Level 1 Activity", command=lambda: switch_frame(last_frame_level1 if last_frame_level1 else frame1), **button_style).pack(pady=5)

# Return to Level 2 (or start at Step 1 if not visited)
tk.Button(nav_frame.content, text="Go to Level 2 Activity", command=lambda: switch_frame(last_frame_level2 if last_frame_level2 else frame7), **button_style).pack(pady=5)

# --- Activity Frames Navigation Button ---
for activity_frame in [frame1, frame2, frame3, frame4, frame5, frame6,
                       frame7, frame8, frame9, frame10, frame11, frame12, frame13]:
    # Menu button
    tk.Button(activity_frame.content, text="☰", command=open_menu, **menu_style).place(x=10, y=10)
    
    # Manual control popup button
    tk.Button(activity_frame.content, text="🛠 Controls", command=show_manual_popup, **menu_style).place(x=60, y=10)

## ACTIVITY LEVEL 1
# --- Frame 1: Setup ---
tk.Label(frame1.center_block, text="LEVEL 1 - STEP 1: SETTING UP", **header_style).pack(pady=20)
tk.Label(frame1.center_block, text="As a teacher, you will guide your students in the process of learning about Reinforcement Learning interactively.", font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack(pady=10)
tk.Label(frame1.center_block, text="• First, setup the Thymio in a place where its arms can move freely.\n• Then, click \"GET STARTED\" and wait.", **label_style).pack(pady=10)
tk.Label(frame1.center_block, text="→ Do not interfere while it's setting up! \n The Thymio will move its arms and setup for about 30 seconds. \n", font=("Microsoft YaHei Light", 14, "italic"), bg="#67d7e6").pack()

tk.Button(frame1.center_block, text="GET STARTED", command=lambda: run_async(lambda: asyncio.run(sts.run_get_started()) or after_get_started()), **button_style).pack()

# --- Frame 2: Front Crawl Discussion ---
tk.Label(frame2.center_block, text="LEVEL 1 - STEP 2: THE FRONT CRAWL", **header_style).pack(pady=20)
tk.Label(frame2.center_block, text="Ask your students:", **label_style).pack(pady=10)
tk.Label(frame2.center_block, text="How did you learn to swim?\nWhat does the front crawl look like?\nWas it hard to learn?\n", font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack(pady=10)
tk.Label(frame2.center_block, text="Tell students they are now Thymio's coach, he doesn't know how to swim yet so they must teach him!\n\nClick \"TEST\" to see what Thymio can do right now.", **label_style).pack(pady=10)
tk.Button(frame2.center_block, text="TEST", command=run_test_then_next, **button_style).pack()

# --- Frame 3: Training ---
tk.Label(frame3.center_block, text="LEVEL 1 - STEP 3: REINFORCEMENT TRAINING", **header_style).pack(pady=20)

tk.Label(frame3.center_block, text="To teach Thymio how to swim, reward him. You can change the amount of times you reward him in the controls section on the top left corner. \n", **label_style).pack()

# Correct movement
tk.Label(frame3.center_block, text='If he moves correctly (arms apart), say "Good job Thymio!" by pressing the button which is:', font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack()
tk.Label(frame3.center_block, text="GREEN", font=("Microsoft YaHei Light", 14, "bold"), fg="green", bg="#67d7e6").pack()

# Incorrect movement (bolded)
tk.Label(frame3.center_block, text='If he moves incorrectly (arms together), say "Oh no!" by pressing the button which is:', font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack()
tk.Label(frame3.center_block, text="RED", font=("Microsoft YaHei Light", 14, "bold"), fg="red", bg="#67d7e6").pack()

tk.Label(frame3.center_block, text="Let students take turns rewarding him.\n\nClick \"TRAIN\" to start.", **label_style).pack()
tk.Label(frame3.center_block, text="At the beginning, Thymio's arms will move to the front, \nthis is normal so that it knows where it starts at.", font=("Microsoft YaHei Light", 14, "italic"), bg="#67d7e6").pack()

tk.Button(frame3.center_block, text="TRAIN", command=run_train_and_continue, **button_style).pack(pady=5)

# --- Frame 4: Testing ---
tk.Label(frame4.center_block, text="LEVEL 1 - STEP 4: TESTING", **header_style).pack(pady=20)
tk.Label(frame4.center_block, text="Now let’s see if Thymio learned to swim!", font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack()
tk.Label(frame4.center_block, text="If students gave the correct rewards during training, Thymio should now swim.\nIf not, his movement may still be uncoordinated. Use the test buttons to observe the results.", **label_style).pack(pady=10)
frame4.correct_photo = correct_photo  # Keep reference

tk.Label(frame4.center_block, text="At the beginning, Thymio's arms will move to the front, this is normal so it knows where it starts.", font=("Microsoft YaHei Light", 14, "italic"), bg="#67d7e6").pack()

test_button_row = tk.Frame(frame4.center_block, bg="#67d7e6")
test_button_row.pack(pady=5)

# START TESTING button
tk.Button(test_button_row, text="START TESTING SWIM",
          command=start_testing, **button_style).pack(side="left", padx=5)

tk.Button(test_button_row, text="STOP TESTING",
          command=stop_testing, **button_style).pack(side="left", padx=5)

tk.Label(frame4.center_block, text="Is Thymio able to swim?", font=("Microsoft YaHei Light", 14, "bold"), bg="#67d7e6", wraplength=600, justify="center").pack(pady=10)
# tk.Label(frame4.center_block, image=correct_photo, bg="#67d7e6").pack(pady=10)

tk.Label(frame4.center_block, text="If yes, great job rewarding him! You can try the next activity if your students know about order of operations and reading a table. If not, you can click \"RESTART LEARNING\" to try again.", **label_style).pack(pady=10)

# Horizontal row for Restart & Start Level 2 buttons
level_buttons_row = tk.Frame(frame4.center_block, bg="#67d7e6")
level_buttons_row.pack(pady=5)

tk.Button(level_buttons_row, text="START LEVEL 2", command=start_activity2, **button_style).pack(side="left", padx=5)
tk.Button(level_buttons_row, text="RESTART LEARNING", command=handle_restart_learning, **button_style).pack(side="left", padx=5)

tk.Label(frame4.center_block, text="If you want to end the activity exit the GUI.", **label_style).pack(pady=10)


## ACTIVITY LEVEL 2
# --- Frame 7: Step 1 – Identify States & Actions ---
tk.Label(frame7.center_block, text="LEVEL 2 - STEP 1: STATES & ACTIONS", **header_style).pack(pady=20)

tk.Label(
    frame7.center_block,
    text="In reinforcement learning, a state is how the world looks at a given moment.\nIn this case, we focus on the position of Thymio’s arms.",
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=(10, 5))

tk.Label(
    frame7.center_block,
    text=(
        "Each arm can be either in the FRONT or BACK, but in different combinations. \n\n"
    ),
    **label_style
).pack(pady=7)

tk.Label(
    frame7.center_block,
    text="Now let’s think about what actions Thymio can take.",
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6"
).pack(pady=(10, 3))

tk.Label(
    frame7.center_block,
    text=(
        "Thymio can:\n"
        "• Move only the right arm or only the left arm\n"
        "• Move both arms at the same time\n\n"
    ),
    **label_style
).pack(pady=3)

tk.Label(
    frame7.center_block,
    text="📄 Students should complete Step 1 in the worksheet before moving on.",
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=5)

tk.Button(frame7.center_block, text="NEXT", command=lambda: switch_frame(frame8), **button_style).pack(pady=10)

# --- Frame 8: Step 2 – Build the Q-table ---
tk.Label(frame8.center_block, text="LEVEL 2 - STEP 2: BUILD THE Q-TABLE", **header_style).pack(pady=20)

tk.Label(
    frame8.center_block,
    text=(
        "Now that students have identified all possible states and actions, it's time to build the Q-table.\n\n"
        "📊 A Q-table is used to track how good it is to take an action from a particular state.\n\n"
        "• Each row represents a state.\n"
        "• Each column represents an action Thymio can take.\n"
        "• At the beginning, all values in the table are zero, which will be updated.\n"
    ),
    **label_style
).pack(pady=10)

tk.Label(
    frame8.center_block,
    text=(
        "📝 Students should now fill in the empty Q-table in Step 2 of their worksheet using the states and actions."
    ),
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=10)

tk.Label(
    frame8.center_block,
    text=(
        "💡 Teacher Tip: As Q-tables grow in size, training takes longer and becomes more complex.\n"
        "This activity uses 4 states and 3 actions to keep it manageable and easy to interpret."
    ),
    **label_style
).pack(pady=10)

tk.Button(frame8.center_block, text="NEXT", command=lambda: switch_frame(frame9), **button_style).pack(pady=10)

# --- Frame 9: Step 3 – Q-value Calculation & Training ---
tk.Label(frame9.center_block, text="LEVEL 2 - STEP 3: CALCULATE Q-VALUES", **header_style).pack(pady=20)

tk.Label(
    frame9.center_block,
    text=(
        "To complete the Q-table, students will now observe Thymio taking actions and apply the Q-learning equation which is simplified with colors and key words in the worksheet.\n\n"
        "Each training step gives us the full information we need:\n"
        "Initial State, Action taken, Reward, Future State\n\n"
        "With this, students can substitute values into the equation and compute the new Q-value."
    ),
    **label_style
).pack(pady=10)

tk.Label(
    frame9.center_block,
    text=(
        "📄 Students should record this process in Step 3 of the worksheet, showing their work.\n"
    ),
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=10)

# Training iteration control
train_control = tk.Frame(frame9.center_block, bg="#67d7e6")
train_control.pack(pady=10)

tk.Label(
    train_control,
    text="Decide how many actions Thymio will perform (Recommended 5):",
    font=("Microsoft YaHei Light", 12, "bold"),
    bg="#67d7e6"
).grid(row=0, column=0, padx=5)

tk.Spinbox(train_control, from_=1, to=100, width=5, textvariable=train_var).grid(row=0, column=1, padx=5)

tk.Label(
    frame9.center_block,
    text="Students will repeat the procedure for Q-table calculation this amount of times so \n" 
    "make sure students have a Q-table worksheet for each training step you plan to run.",
    **label_style
).pack(pady=10)

tk.Button(frame9.center_block, text="NEXT", command=lambda: switch_frame(frame10), **button_style).pack(pady=10)

# --- Frame 10: Level 2 Training ---
tk.Label(
    frame10.center_block,
    text="LEVEL 2 - STEP 3: STEP-BY-STEP TRAINING",
    **header_style
).pack(pady=20)

tk.Label(
    frame10.center_block,
    text=(
        "Click the \"TRAIN\" button to make Thymio perform one action.\n"
        "Immediately after, the screen will display the key information students need:"
        "\n• Initial state\n• Action taken\n• New state\n• Correct reward (expected for front crawl).\n"
        "Unlike level 1, you do not have to give the reward, the correct reward will be given automatically."
    ),
    **label_style
).pack(pady=10)

train10_frame = frame10.center_block

train_btn = tk.Button(train10_frame, text="TRAIN", command=run_level2_step, **button_style)
train_btn.pack(pady=5)

reward_button = tk.Button(train10_frame, text="GIVE REWARD", command=reward_and_continue_level2, **button_style)
# This button is revealed dynamically after action

tk.Label(
    frame10.center_block,
    text=(
        "📝 After observing the movement, guide students to answer Step 3 in the Worksheet\n"
    ),
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=10)


# --- Frame 12: Level 2 Testing ---
tk.Label(frame12.center_block, text="LEVEL 2 - STEP 4: FINAL TEST", **header_style).pack(pady=20)

tk.Label(
    frame12.center_block,
    text=(
        "If fewer than 10 iterations were completed, Thymio most likely did not learn to swim effectively due to limited feedback.\n\n"
        "You can see this by starting testing swim or you can now run virtual training, where the Q-table is filled automatically by the computer. "
        "This simulates many training steps instantly without requiring further interaction.\n\n"
        "This is how machine learning is typically done: fast, scalable, and efficient.\n\n"
        "Try adding virtual training below to improve Thymio’s learning rate."
    ),
    **label_style
).pack(pady=10)

# Pretraining slider + label
pretrain_section = tk.Frame(frame12.center_block, bg="#67d7e6")

tk.Label(
    pretrain_section,
    text="💻 Virtual training iterations (recommended: 30):",
    font=("Microsoft YaHei Light", 12, "bold"),
    bg="#67d7e6"
).grid(row=0, column=0, padx=5)

tk.Spinbox(
    pretrain_section,
    from_=1,
    to=100,
    width=5,
    textvariable=pretrain_var
).grid(row=0, column=1, padx=5)

# 🧠 Pack the section so it appears!
pretrain_section.pack(pady=10)

# Virtual training button
tk.Button(
    frame12.center_block,
    text="VIRTUAL TRAINING",
    command=run_virtual_training_with_popup,
    **button_style
).pack(pady=10)

tk.Label(
    frame12.center_block,
    text="Now let’s see if Thymio learned from building the Q-table!",
    font=("Microsoft YaHei Light", 14, "bold"),
    bg="#67d7e6",
    wraplength=600,
    justify="center"
).pack(pady=5)

# Final testing buttons (continuous loop)
test_button_row = tk.Frame(frame12.center_block, bg="#67d7e6")
test_button_row.pack(pady=5)

# START TESTING button
tk.Button(test_button_row, text="START TESTING SWIM",
          command=start_testing, **button_style).pack(side="left", padx=5)

tk.Button(test_button_row, text="STOP TESTING",
          command=stop_testing, **button_style).pack(side="left", padx=5)

# --- Frame 13: Level 2 Reflection ---
tk.Label(frame13.center_block, text="LEVEL 2 - STEP 5: REFLECTION", **header_style).pack(pady=20)

# 📄 Student reflection questions
reflection_text = (
    "📘 Questions:\n\n"
    "1. Which action has the highest value in each state? Why do you think that is?\n\n"
    "2. What would happen if we continued filling in the Q-table for many more steps?\n\n"
    "3. How does a positive reward change the valu es in the table?\n\n"
    "4. What are the main steps needed to build and fill in a Q-table?\n\n"
    "5. How does this help Thymio 'learn' to swim?"
)

tk.Label(
    frame13.center_block,
    text=reflection_text,
    **label_style
).pack(pady=10)

# ✅ Return to menu
tk.Button(
    frame13.center_block,
    text="RETURN TO HOME",
    command=lambda: [sts.restart_learning(), switch_frame(mode_select_frame)],
    **button_style
).pack(pady=15)

# --- Frame Online Training ---
tk.Label(online_training_frame.center_block, text="VIRTUAL TRAINING", **header_style).pack(pady=20)

tk.Label(online_training_frame.center_block,
         text="This mode lets Thymio learn to swim using computer-based logic.\n"
              "No physical reward-giving is needed.\n\n"
              "During Virtual Training, Thymio fills its Q-table by simulating feedback internally, therefore you won't see any movement.\n"
              "You can choose how many iterations to run to improve accuracy. The higher this number the more likely you will achieve the desired behavior (Recommended: 30)",
         **label_style).pack(pady=10)

train_control = tk.Frame(online_training_frame.center_block, bg="#67d7e6")
train_control.pack(pady=10)

tk.Label(train_control, text="Training Iterations:", font=("Microsoft YaHei Light", 12, "bold"), bg="#67d7e6").grid(row=0, column=0, padx=5)
tk.Spinbox(train_control, from_=1, to=100, width=5, textvariable=pretrain_var).grid(row=0, column=1, padx=5)

def run_virtual_training_and_continue():
    global has_initialized_thymio
    if not has_initialized_thymio:
        error_popup = tk.Toplevel(root)
        error_popup.title("Thymio Not Initialized")
        error_popup.configure(bg="#f9f9f9")

        tk.Label(
            error_popup,
            text="⚠️ Please run GET STARTED before virtual training! It takes 30 seconds to initialize. You should see the Thymio move.",
            font=("Segoe UI", 12, "bold"),
            bg="#f9f9f9",
            wraplength=300
        ).pack(padx=20, pady=20)

        tk.Button(
            error_popup,
            text="Run GET STARTED",
            command=lambda: [
                run_async(lambda: asyncio.run(sts.run_get_started()) or set_initialized()),
                error_popup.destroy()
            ],
            bg="#388994",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=5
        ).pack(pady=10)
    else:
        sts.run_pretrain(pretrain_var.get())
        switch_frame(virtual_done_frame)

tk.Button(online_training_frame.center_block, text="START VIRTUAL TRAINING",
          command=run_virtual_training_and_continue, **button_style).pack(pady=10)

tk.Button(online_training_frame.center_block, text="← BACK TO HOME", command=lambda: switch_frame(mode_select_frame), **button_style).pack(pady=10)

# --- Virtual Training Done ---
tk.Label(virtual_done_frame.center_block, text="VIRTUAL TRAINING COMPLETE", **header_style).pack(pady=20)

# Updated instructional text
tk.Label(virtual_done_frame.center_block,
         text="Online training is complete.\n\nYou can now test Thymio's movements to see if it is able to swim.\nClick START TESTING to begin observing Thymio's behavior.",
         **label_style).pack(pady=10)

# Correct behavior popup button
tk.Button(virtual_done_frame.content, text="Correct behavior ❓", command=show_testing_image_popup, **menu_style).place(x=800, y=5)

# Frame for testing buttons
testing_buttons_row = tk.Frame(virtual_done_frame.center_block, bg="#67d7e6")
testing_buttons_row.pack(pady=10)

# Define testing buttons
def start_testing_with_check():
    global has_initialized_thymio
    if not has_initialized_thymio:
        error_popup = tk.Toplevel(root)
        error_popup.title("Thymio Not Initialized")
        error_popup.configure(bg="#f9f9f9")

        tk.Label(
            error_popup,
            text="⚠️ Please run GET STARTED before testing and try again! It takes 30 seconds to initialize. You should see the Thymio move.",
            font=("Segoe UI", 12, "bold"),
            bg="#f9f9f9",
            wraplength=300
        ).pack(padx=20, pady=20)

        tk.Button(
            error_popup,
            text="Run GET STARTED",
            command=lambda: [
                run_async(lambda: asyncio.run(sts.run_get_started()) or set_initialized()),
                error_popup.destroy()
            ],
            bg="#388994",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=5
        ).pack(pady=10)
    else:
        start_testing()

# Define testing buttons using the check function
tk.Button(testing_buttons_row, text="START TESTING",
          command=start_testing_with_check, **button_style).pack(side="left", padx=5)

tk.Button(testing_buttons_row, text="STOP TESTING",
          command=stop_testing, **button_style).pack(side="left", padx=5)

# Return to Home button
tk.Button(virtual_done_frame.center_block, text="RETURN TO HOME",
          command=lambda: switch_frame(mode_select_frame),
          **button_style).pack(pady=15)

# --- Start GUI ---
switch_frame(intro_frame)
root.mainloop()