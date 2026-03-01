# Setup Guide: From Zero to Running

This guide assumes you have **never** used a terminal, Git, or Python before. Follow every step and you'll be running code in 15 minutes.

!!! success "Don't want to install anything?"
    You can read all chapters and run Python code **right now** in your browser:

    - [Read chapters on this website](../chapters/index.md) -- no setup needed
    - [Run code in the Playground](../playground.md) -- Python runs in your browser

    The instructions below are for people who want to work **locally on their computer**, which gives you the full experience with Jupyter notebooks.

---

## Step 0: What You Need

- A computer (Windows, Mac, or Linux)
- An internet connection
- 15 minutes

That's it. Everything else we'll install together.

---

## Step 1: Install Python

Python is the programming language used in every chapter.

=== "Windows"

    1. Open your web browser
    2. Go to [python.org/downloads](https://www.python.org/downloads/)
    3. Click the big yellow **"Download Python 3.x.x"** button
    4. Run the downloaded file
    5. **IMPORTANT**: Check the box that says **"Add Python to PATH"** at the bottom of the installer. This is the most common mistake beginners make.
    6. Click **"Install Now"**
    7. Wait for it to finish, then click **"Close"**

    !!! warning "Don't forget: Add Python to PATH"
        If you skip this checkbox, nothing else in this guide will work.
        If you already installed without it, uninstall Python and reinstall with the box checked.

=== "Mac"

    1. Open your web browser
    2. Go to [python.org/downloads](https://www.python.org/downloads/)
    3. Click **"Download Python 3.x.x"**
    4. Open the downloaded `.pkg` file
    5. Follow the installer (click Continue, Agree, Install)
    6. Done

=== "Linux"

    Python is usually pre-installed. Open a terminal and type:

    ```
    python3 --version
    ```

    If you see a version number (3.10 or higher), you're good. If not:

    ```
    sudo apt update && sudo apt install python3 python3-pip python3-venv
    ```

### Verify Python is installed

=== "Windows"

    1. Press the **Windows key** on your keyboard
    2. Type **PowerShell**
    3. Click **"Windows PowerShell"** (the blue icon)
    4. In the window that opens, type this and press Enter:

    ```
    python --version
    ```

    You should see something like `Python 3.12.3`. If you see an error, Python wasn't added to PATH -- reinstall it.

=== "Mac"

    1. Press **Cmd + Space** to open Spotlight
    2. Type **Terminal** and press Enter
    3. In the window that opens, type:

    ```
    python3 --version
    ```

    You should see something like `Python 3.12.3`.

=== "Linux"

    Open a terminal and type:

    ```
    python3 --version
    ```

---

## Step 2: Install Git

Git lets you download the chapters to your computer.

=== "Windows"

    1. Go to [git-scm.com/download/win](https://git-scm.com/download/win)
    2. The download should start automatically
    3. Run the installer
    4. Click **Next** on every screen (the defaults are fine)
    5. Click **Install**, then **Finish**

=== "Mac"

    Open Terminal and type:

    ```
    git --version
    ```

    If Git isn't installed, your Mac will ask if you want to install developer tools. Click **Install** and wait.

=== "Linux"

    ```
    sudo apt install git
    ```

### Verify Git is installed

Close your terminal/PowerShell and open a **new** one, then type:

```
git --version
```

You should see something like `git version 2.43.0`.

---

## Step 3: Download the Chapters

Now we'll download all the Berta Chapters to your computer.

=== "Windows (PowerShell)"

    1. Open **PowerShell** (press Windows key, type PowerShell, click it)
    2. Type these commands one at a time, pressing Enter after each:

    ```powershell
    cd Desktop
    git clone https://github.com/luigipascal/berta-chapters.git
    cd berta-chapters
    ```

    This creates a folder called `berta-chapters` on your Desktop.

=== "Mac (Terminal)"

    1. Open **Terminal**
    2. Type these commands one at a time:

    ```bash
    cd Desktop
    git clone https://github.com/luigipascal/berta-chapters.git
    cd berta-chapters
    ```

=== "Linux (Terminal)"

    ```bash
    cd ~
    git clone https://github.com/luigipascal/berta-chapters.git
    cd berta-chapters
    ```

!!! info "What just happened?"
    `cd Desktop` moved you to your Desktop folder.
    `git clone ...` downloaded all the chapters from GitHub to your computer.
    `cd berta-chapters` moved you into the downloaded folder.

---

## Step 4: Install Dependencies

The chapters need some Python libraries to work. Install them all at once:

=== "Windows"

    ```powershell
    pip install -r requirements.txt
    ```

=== "Mac / Linux"

    ```bash
    pip3 install -r requirements.txt
    ```

This takes 1-2 minutes. You'll see a lot of text scrolling by -- that's normal.

!!! tip "If you get a permission error"
    Try adding `--user` at the end:

    ```
    pip install -r requirements.txt --user
    ```

---

## Step 5: Open the Chapters

You have two options: **Jupyter Notebook** (classic) or **VS Code** (modern editor).

### Option A: Jupyter Notebook (Recommended for Beginners)

Jupyter Notebook opens in your web browser and lets you run code one cell at a time.

=== "Windows"

    ```powershell
    cd chapters\chapter-01-python-fundamentals
    jupyter notebook
    ```

=== "Mac / Linux"

    ```bash
    cd chapters/chapter-01-python-fundamentals
    jupyter notebook
    ```

Your web browser will open automatically. You'll see a list of files. Click on the `notebooks` folder, then click `01_introduction.ipynb`.

!!! info "How to use Jupyter Notebook"
    - Click on a cell (the gray boxes with code) to select it
    - Press **Shift + Enter** to run the selected cell
    - The output appears below the cell
    - Work through cells from top to bottom
    - You can edit the code and re-run cells to experiment

To stop Jupyter, go back to your terminal and press **Ctrl + C**.

### Option B: VS Code (Recommended for Developers)

VS Code is a free code editor with built-in Jupyter support.

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install it (run the downloaded file, click Next until done)
3. Open VS Code
4. Click **File > Open Folder**
5. Navigate to Desktop > berta-chapters and click **Select Folder**
6. VS Code will ask to install recommended extensions -- click **Yes**
7. In the left sidebar, navigate to `chapters/chapter-01-python-fundamentals/notebooks/`
8. Click on `01_introduction.ipynb`
9. VS Code will ask to install the Jupyter extension -- click **Install**
10. Click the **Run All** button at the top, or click the play button next to each cell

---

## Step 6: Start Learning

You're set up! Here's what to do now:

1. **Work through Chapter 1** -- start with `01_introduction.ipynb`
2. **Run every code cell** -- don't just read, experiment!
3. **Try the exercises** -- in the `exercises/` folder
4. **When you finish Chapter 1**, move to Chapter 2:

=== "Windows"

    ```powershell
    cd ..\chapter-02-data-structures
    jupyter notebook
    ```

=== "Mac / Linux"

    ```bash
    cd ../chapter-02-data-structures
    jupyter notebook
    ```

---

## Getting Updates

When new chapters are released, get them by running this in the `berta-chapters` folder:

```
git pull
```

This downloads any new or updated chapters.

---

## Troubleshooting

### "python is not recognized" (Windows)

Python wasn't added to PATH. Uninstall Python, then reinstall and check the **"Add Python to PATH"** box.

### "pip is not recognized" (Windows)

Same fix as above. Reinstall Python with the PATH checkbox.

### "jupyter is not recognized"

Jupyter didn't install correctly. Try:

```
pip install jupyter notebook
```

### "Permission denied"

Add `--user` to pip commands:

```
pip install -r requirements.txt --user
```

### Jupyter opens but shows an error in the notebook

Make sure you installed the dependencies (Step 4). Go back to the `berta-chapters` folder and run `pip install -r requirements.txt` again.

### The terminal shows weird characters or colors

That's normal -- some tools use colors in the terminal. Everything is working fine.

### Nothing works and I'm stuck

No worries! You can still learn without any local setup:

- **[Read chapters on the website](../chapters/index.md)** -- all content is here
- **[Use the Playground](../playground.md)** -- run Python in your browser
- **[Open a discussion](https://github.com/luigipascal/berta-chapters/discussions)** -- ask for help

---

## Quick Reference Card

| Task | Windows (PowerShell) | Mac / Linux (Terminal) |
|------|---------------------|----------------------|
| Open terminal | Windows key > type "PowerShell" | Cmd+Space > type "Terminal" |
| Go to Desktop | `cd Desktop` | `cd Desktop` |
| Download chapters | `git clone https://github.com/luigipascal/berta-chapters.git` | same |
| Enter folder | `cd berta-chapters` | same |
| Install libraries | `pip install -r requirements.txt` | `pip3 install -r requirements.txt` |
| Open Chapter 1 | `cd chapters\chapter-01-python-fundamentals` | `cd chapters/chapter-01-python-fundamentals` |
| Start Jupyter | `jupyter notebook` | same |
| Stop Jupyter | Ctrl + C in terminal | same |
| Get updates | `git pull` | same |

---

**Created by [Luigi Pascal Rondanini](https://rondanini.net) | Generated by [Berta AI](https://berta.one)**
