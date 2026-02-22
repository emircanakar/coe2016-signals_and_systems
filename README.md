# COE 216 - Signals and Systems (Spring 2026)

[cite_start]This repository serves as a central hub for all homework assignments and projects for the **COE 216 Signals and Systems** course at **Istanbul Health and Technology University (İSTÜN)** [cite: 89-91, 93, 94].

## 👥 Group Members
* [cite_start]**220611010** - Sübhan Aslan [cite: 98]
* [cite_start]**230611055** - Osman Bay [cite: 99]
* [cite_start]**240611303** - Emircan Akar [cite: 100]

---

## 📂 Repository Structure
Assignments are organized into dedicated folders to ensure clarity and easy navigation throughout the semester.

### [cite_start][Homework 1: Sinusoidal Signals & DTMF Synthesis](https://github.com/your-username/coe2016-signals_and_systems/tree/main/Homework_1) [cite: 3]
* **Task 1: Sampling and Visualization**: 
    * [cite_start]Calculation of baseline frequency $f_{0} = 68 \text{ Hz}$ based on group school numbers[cite: 113].
    * [cite_start]Visualization of $f_{1}=68 \text{ Hz}$, $f_{2}=34 \text{ Hz}$, and $f_{3}=680 \text{ Hz}$ sinusoidal signals[cite: 115].
    * [cite_start]Demonstration of the Nyquist-Shannon Sampling Theorem by selecting $f_{s} = 10,000 \text{ Hz}$ [cite: 117-120].
* **Task 2: DTMF Interface Design**:
    * [cite_start]Interactive phone keypad (numpad) design including 0-9, *, #, and A-D keys[cite: 53].
    * [cite_start]Real-time time-domain signal visualization upon keypress[cite: 55].
    * [cite_start]Dual-tone audio synthesis using frequency summation: $x(t) = \sin(2\pi f_{low}t) + \sin(2\pi f_{high}t)$[cite: 48, 54, 190].
    * [cite_start]Signal normalization by a factor of 0.5 to prevent digital clipping[cite: 75, 193].

---

## 🛠 Installation & Requirements
The projects in this repository are developed using **Python**. To run the scripts, you must install the following dependencies:

```bash
pip install numpy matplotlib sounddevice

NumPy: Used for mathematical signal generation and array operations.
Matplotlib: Utilized for signal visualization in the time domain.
Sounddevice: Employed for real-time audio synthesis.
Tkinter: Used for the interactive Graphical User Interface (GUI).

Usage
Navigate to the specific homework folder and execute the tasks:
# To run Homework 1 tasks
python task-1.py
python task-2.py

Course Information

Lecturer: Prof. Halis Altun 
Department: Computer Engineering 
Semester: 2025-2026 Spring

Academic Integrity
All materials in this repository are submitted for academic purposes. The implementations and reports are the original works of the listed group members, complying with the university's academic policies.
