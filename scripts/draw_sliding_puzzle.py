#!/usr/bin/env python3
"""
Create a figure illustrating the 2x3 sliding puzzle and its three moves.

Top: board layout + move definitions (V, L, R) with intermediate steps for L/R.
Bottom: example trajectory from start to goal.
"""

import sys
import os
NOTES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(NOTES_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'quconet'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from quconet.slidingpuzzle import (
    WINNING_STATE, apply_move,
    create_sliding_puzzle_graph,
)

# Tile colours
TILE_COLORS = {
    0: '#f0f0f0', 1: '#4E79A7', 2: '#F28E2B', 3: '#E15759',
    4: '#76B7B2', 5: '#59A14F',
}
TEXT_COLORS = {0: '#aaaaaa', 1: 'white', 2: 'white', 3: 'white',
               4: 'white', 5: 'white'}


def draw_puzzle_at(ax, state, x0, y0, cell=0.38, highlight_positions=None,
                   label=None, label_below=None):
    """Draw a 2x3 puzzle board at (x0, y0) with optional position highlights.

    highlight_positions: list of board indices (0-5) to draw with a
        coloured border (for showing which tiles move).
    """
    pad = 0.03
    for idx, val in enumerate(state):
        col = idx % 3
        row = 1 - idx // 3
        x = x0 + col * cell
        y = y0 + row * cell

        ec = '#444444'
        lw = 0.8
        if highlight_positions and idx in highlight_positions:
            ec = '#c03030'
            lw = 2.0

        rect = mpatches.FancyBboxPatch(
            (x + pad, y + pad), cell - 2 * pad, cell - 2 * pad,
            boxstyle='round,pad=0.03',
            facecolor=TILE_COLORS[val], edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        txt = str(val) if val != 0 else ''
        ax.text(x + cell / 2, y + cell / 2, txt,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=TEXT_COLORS[val])

    w = 3 * cell
    h = 2 * cell
    if label:
        ax.text(x0 + w / 2, y0 + h + 0.08, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    if label_below:
        ax.text(x0 + w / 2, y0 - 0.12, label_below,
                ha='center', va='top', fontsize=7, color='#555555')


def draw_arrow(ax, x1, y1, x2, y2, label=None, label_below=None, color='#333333'):
    """Draw a horizontal arrow with optional labels."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                mutation_scale=14))
    if label:
        mx = (x1 + x2) / 2
        ax.text(mx, y2 + 0.15, label, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#c03030')
    if label_below:
        mx = (x1 + x2) / 2
        ax.text(mx, y2 - 0.15, label_below, ha='center', va='top',
                fontsize=6, color='#888888')


def draw_small_arrow(ax, x1, y1, x2, y2, color='#999999'):
    """Small arrow for intermediate steps."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0,
                                mutation_scale=10))


def do_single_swap(state, pos_a, pos_b):
    """Swap two positions, return new state."""
    s = list(state)
    s[pos_a], s[pos_b] = s[pos_b], s[pos_a]
    return s


def main():
    G, state_to_index, index_to_state = create_sliding_puzzle_graph()

    cell = 0.38
    bw = 3 * cell   # board width
    bh = 2 * cell   # board height
    gap = 0.55      # gap between boards in move demos
    small_gap = 0.35  # gap for intermediate steps

    # ── Layout ──
    fig_w = 13.5
    fig_h = 6.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Move definitions: 2 rows ──
    # Row 1: V↑ (hole moves up) + L (left cycle)
    # Row 2: V↓ (hole moves down) + R (right cycle)

    row1_y = 4.6
    row2_y = 3.0

    # Compute total row width to centre everything
    # V section: 2 boards + gap
    v_section_w = 2 * bw + gap
    # L/R section: 4 boards + 3 small_gaps
    lr_section_w = 4 * bw + 3 * small_gap
    # Spacing between V and L/R sections
    section_gap = 0.8
    total_row_w = v_section_w + section_gap + lr_section_w
    row_x0 = (fig_w - 0.5 - total_row_w) / 2  # centre offset

    v_x = row_x0
    lr_x = row_x0 + v_section_w + section_gap

    # -- Row 1: V↑ + L --

    # V↑: hole at bottom centre (pos 4), swaps up to pos 1
    vu_state = [1, 2, 3, 4, 0, 5]  # hole at pos 4
    vu_after = do_single_swap(vu_state, 1, 4)

    draw_puzzle_at(ax, vu_state, v_x, row1_y, cell, highlight_positions=[1, 4])
    draw_arrow(ax, v_x + bw + 0.08, row1_y + bh / 2,
               v_x + bw + gap - 0.08, row1_y + bh / 2, label='V↑')
    draw_puzzle_at(ax, vu_after, v_x + bw + gap, row1_y, cell,
                   highlight_positions=[1, 4])
    ax.text(v_x + bw + gap / 2, row1_y + bh + 0.3,
            'V (up): swap centres',
            ha='center', fontsize=9, fontweight='bold', color='#333333')

    # L: hole at pos 1, rotates 1→0→3→4
    l_state = [3, 0, 2, 1, 5, 4]
    l_step1 = do_single_swap(l_state, 1, 0)
    l_step2 = do_single_swap(l_step1, 0, 3)
    l_step3 = do_single_swap(l_step2, 3, 4)
    l_states = [l_state, l_step1, l_step2, l_step3]
    l_highlights = [[0, 1], [0, 3], [3, 4], [1, 4]]

    for i, (st, hl) in enumerate(zip(l_states, l_highlights)):
        x = lr_x + i * (bw + small_gap)
        draw_puzzle_at(ax, st, x, row1_y, cell, highlight_positions=hl)
        if i < 3:
            draw_small_arrow(ax,
                             x + bw + 0.05, row1_y + bh / 2,
                             x + bw + small_gap - 0.05, row1_y + bh / 2)

    ax.text(lr_x + (3 * (bw + small_gap) + bw) / 2, row1_y + bh + 0.3,
            'L (left cycle): hole rotates through left corners',
            ha='center', fontsize=9, fontweight='bold', color='#333333')
    brace_x1, brace_x2 = lr_x, lr_x + 3 * (bw + small_gap) + bw
    ax.annotate('', xy=(brace_x2, row1_y - 0.18), xytext=(brace_x1, row1_y - 0.18),
                arrowprops=dict(arrowstyle='<->', color='#c03030', lw=1.2))
    ax.text((brace_x1 + brace_x2) / 2, row1_y - 0.30,
            '= one move L', ha='center', fontsize=7, color='#c03030')

    # -- Row 2: V↓ + R --

    # V↓: hole at top centre (pos 1), swaps down to pos 4
    vd_state = [1, 0, 3, 4, 2, 5]  # hole at pos 1
    vd_after = do_single_swap(vd_state, 1, 4)

    draw_puzzle_at(ax, vd_state, v_x, row2_y, cell, highlight_positions=[1, 4])
    draw_arrow(ax, v_x + bw + 0.08, row2_y + bh / 2,
               v_x + bw + gap - 0.08, row2_y + bh / 2, label='V↓')
    draw_puzzle_at(ax, vd_after, v_x + bw + gap, row2_y, cell,
                   highlight_positions=[1, 4])
    ax.text(v_x + bw + gap / 2, row2_y + bh + 0.3,
            'V (down): swap centres',
            ha='center', fontsize=9, fontweight='bold', color='#333333')

    # R: hole at pos 1, rotates 1→2→5→4
    r_state = [3, 0, 2, 1, 5, 4]
    r_step1 = do_single_swap(r_state, 1, 2)
    r_step2 = do_single_swap(r_step1, 2, 5)
    r_step3 = do_single_swap(r_step2, 5, 4)
    r_states = [r_state, r_step1, r_step2, r_step3]
    r_highlights = [[1, 2], [2, 5], [4, 5], [1, 4]]

    for i, (st, hl) in enumerate(zip(r_states, r_highlights)):
        x = lr_x + i * (bw + small_gap)
        draw_puzzle_at(ax, st, x, row2_y, cell, highlight_positions=hl)
        if i < 3:
            draw_small_arrow(ax,
                             x + bw + 0.05, row2_y + bh / 2,
                             x + bw + small_gap - 0.05, row2_y + bh / 2)

    ax.text(lr_x + (3 * (bw + small_gap) + bw) / 2, row2_y + bh + 0.3,
            'R (right cycle): hole rotates through right corners',
            ha='center', fontsize=9, fontweight='bold', color='#333333')
    brace_x1, brace_x2 = lr_x, lr_x + 3 * (bw + small_gap) + bw
    ax.annotate('', xy=(brace_x2, row2_y - 0.18), xytext=(brace_x1, row2_y - 0.18),
                arrowprops=dict(arrowstyle='<->', color='#c03030', lw=1.2))
    ax.text((brace_x1 + brace_x2) / 2, row2_y - 0.30,
            '= one move R', ha='center', fontsize=7, color='#c03030')

    # ── Divider ──
    ax.plot([1.6, 11.4], [2.42, 2.42], color='#dddddd', lw=0.8)

    # ── Bottom section: Example trajectory ──
    bot_y = 1.0
    traj_cell = 0.42
    traj_bw = 3 * traj_cell
    traj_gap = 0.7

    # Generate trajectory
    random.seed(42)
    goal_state = WINNING_STATE.copy()
    moves_pool = ['V', 'L', 'R']
    random_moves = []
    prev_move = None
    for _ in range(4):
        available = [m for m in moves_pool if m != prev_move]
        move = random.choice(available)
        random_moves.append(move)
        prev_move = move

    scrambled = goal_state.copy()
    forward = [scrambled.copy()]
    for move in random_moves:
        scrambled = apply_move(scrambled, move)
        forward.append(scrambled.copy())

    solution_moves = list(reversed(random_moves))
    trajectory = [forward[-1].copy()]
    current = forward[-1].copy()
    for move in solution_moves:
        current = apply_move(current, move)
        trajectory.append(current.copy())

    node_indices = [state_to_index[tuple(s)] for s in trajectory]
    labels = ['start', 'step 1', 'step 2', 'step 3', 'goal']
    MOVE_LONG = {'V': 'vertical', 'L': 'left cycle', 'R': 'right cycle'}

    # Center the trajectory
    n_states = len(trajectory)
    total_traj_w = n_states * traj_bw + (n_states - 1) * traj_gap
    traj_x0 = (fig_w - 0.5 - total_traj_w) / 2

    for i in range(n_states):
        x = traj_x0 + i * (traj_bw + traj_gap)
        draw_puzzle_at(ax, trajectory[i], x, bot_y, traj_cell,
                       label=labels[i],
                       label_below=f'node {node_indices[i]}')

        if i < n_states - 1:
            mv = solution_moves[i]
            ax1 = x + traj_bw + 0.08
            ax2 = x + traj_bw + traj_gap - 0.08
            my = bot_y + traj_cell  # mid height
            draw_arrow(ax, ax1, my, ax2, my, label=mv,
                       label_below=MOVE_LONG[mv])

    # Highlight goal
    gx = traj_x0 + (n_states - 1) * (traj_bw + traj_gap)
    bg = mpatches.FancyBboxPatch(
        (gx - 0.04, bot_y - 0.04),
        traj_bw + 0.08, 2 * traj_cell + 0.08,
        boxstyle='round,pad=0.04', facecolor='none',
        edgecolor='#d4a017', linewidth=2.0)
    ax.add_patch(bg)

    ax.text(traj_x0 + total_traj_w / 2, bot_y - 0.45,
            'Example: 4-step solution on the $2{\\times}3$ sliding puzzle  '
            '($N{=}120$ nodes, $K{=}3$ moves)',
            ha='center', fontsize=9, color='#555555')

    # ── Final formatting ──
    ax.set_xlim(0.0, fig_w - 0.5)
    ax.set_ylim(0.25, row1_y + bh + 0.62)

    out_pdf = os.path.join(NOTES_DIR, 'figs', 'Q_task_sliding_puzzle.pdf')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_pdf}')
    plt.close()


if __name__ == '__main__':
    main()
