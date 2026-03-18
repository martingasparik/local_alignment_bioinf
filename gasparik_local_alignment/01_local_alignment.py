import numpy as np

# used for Amino Acids: https://web.expasy.org/cgi-bin/sim/sim.pl?prot
# used for DNA: https://www.ebi.ac.uk/jdispatcher/psa/emboss_water?stype=dna&matrix=EDNAFULL

FILE1      = "mouse.fasta"
FILE2      = "human.fasta"
GAP_OPEN   = - 9.5
GAP_EXTEND = - 0.5

EDNAFULL = """
   A  T  G  C  N
A  5 -4 -4 -4 -2
T -4  5 -4 -4 -2
G -4 -4  5 -4 -2
C -4 -4 -4  5 -2
N -2 -2 -2 -2 -1
"""

BLOSUM62_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

PAM250_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  2 -2  0  0 -2  0  0  1 -1 -1 -2 -1 -1 -3  1  1  1 -6 -3  0
R -2  6  0 -1 -4  1 -1 -3  2 -2 -3  3  0 -4  0  0 -1  2 -4 -2
N  0  0  2  2 -4  1  1  0  2 -2 -3  1 -2 -3  0  1  0 -4 -2 -2
D  0 -1  2  4 -5  2  3  1  1 -2 -4  0 -3 -6 -1  0  0 -7 -4 -2
C -2 -4 -4 -5 12 -5 -5 -3 -3 -2 -6 -5 -5 -4 -3  0 -2 -8  0 -2
Q  0  1  1  2 -5  4  2 -1  3 -2 -2  1 -1 -5  0 -1 -1 -5 -4 -2
E  0 -1  1  3 -5  2  4  0  1 -2 -3  0 -2 -5 -1  0  0 -7 -4 -2
G  1 -3  0  1 -3 -1  0  5 -2 -3 -4 -2 -3 -5  0  1  0 -7 -5 -1
H -1  2  2  1 -3  3  1 -2  6 -2 -2  0 -2 -2  0 -1 -1 -3  0 -2
I -1 -2 -2 -2 -2 -2 -2 -3 -2  5  2 -2  2  1 -2 -1  0 -5 -1  4
L -2 -3 -3 -4 -6 -2 -3 -4 -2  2  6 -3  4  2 -3 -3 -2 -2 -1  2
K -1  3  1  0 -5  1  0 -2  0 -2 -3  5  0 -5 -1  0  0 -3 -4 -2
M -1  0 -2 -3 -5 -1 -2 -3 -2  2  4  0  6  0 -2 -2 -1 -4 -2  2
F -3 -4 -3 -6 -4 -5 -5 -5 -2  1  2 -5  0  9 -5 -3 -3  0  7 -1
P  1  0  0 -1 -3  0 -1  0  0 -2 -3 -1 -2 -5  6  1  0 -6 -5 -1
S  1  0  1  0  0 -1  0  1 -1 -2 -3  0 -2 -3  1  2  1 -2 -3 -1
T  1 -1  0  0 -2 -1  0  0 -1  0 -2  0 -1 -3  0  1  3 -5 -3  0
W -6  2 -4 -7 -8 -5 -7 -7 -3 -5 -2 -3 -4  0 -6 -2 -5 17  0 -6
Y -3 -4 -2 -4  0 -4 -4 -5  0 -1 -1 -4 -2  7 -5 -3 -3  0 10 -2
V  0 -2 -2 -2 -2 -2 -2 -1 -2  4  2 -2  2 -1 -1 -1  0 -6 -2  4
"""


def ask_user():
    print("1) DNA")
    print("2) Amino Acid Sequence")
    mode = input("Choose (1/2): ").strip()
    mat_choice = None
    if mode == "2":
        print("1) BLOSUM62")
        print("2) PAM250")
        mat_choice = input("Choose Matrix (1/2): ").strip()
    return mode, mat_choice


def pase_matrix(raw):
    lines = [l for l in raw.strip().split("\n") if l.strip()]
    headers = lines[0].split()
    mat = {}
    for line in lines[1:]:
        parts = line.split()
        aa = parts[0]
        mat[aa] = {}
        for h, s in zip(headers, map(int, parts[1:])):
            mat[aa][h] = s
    return mat


def get_matrix(mode, mat_choice):
    if mode == "1":
        return pase_matrix(EDNAFULL), "EDNAFULL"
    elif mat_choice == "2":
        return pase_matrix(PAM250_RAW), "PAM250"
    else:
        return pase_matrix(BLOSUM62_RAW), "BLOSUM62"


def score(a, b, sub_matrix):
    return sub_matrix[a][b]

def read_fasta(path):
    seq = []
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith(">"):
                    seq.append(line.strip().upper())
        return "".join(seq)
    except FileNotFoundError:
        return ""


def smith_waterman(seq1, seq2, sub_matrix):
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1), dtype=float)
    trace_matrix = np.zeros((m + 1, n + 1), dtype=int)
    E = np.full((m + 1, n + 1), -np.inf)
    F = np.full((m + 1, n + 1), -np.inf)
    max_score = 0
    max_pos   = (0, 0)
    STOP, DIAG, UP, LEFT = 0, 1, 2, 3

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            E[i][j] = max(score_matrix[i-1][j] + GAP_OPEN + GAP_EXTEND,
                          E[i-1][j] + GAP_EXTEND)
            F[i][j] = max(score_matrix[i][j-1] + GAP_OPEN + GAP_EXTEND,
                          F[i][j-1] + GAP_EXTEND)

            match  = score_matrix[i-1][j-1] + score(seq1[i-1], seq2[j-1], sub_matrix)
            delete = E[i][j]
            insert = F[i][j]
            best   = max(0, match, delete, insert)
            score_matrix[i][j] = best

            if   best == 0:      trace_matrix[i][j] = STOP
            elif best == delete: trace_matrix[i][j] = UP
            elif best == insert: trace_matrix[i][j] = LEFT
            else:                trace_matrix[i][j] = DIAG

            if best > max_score:
                max_score = best
                max_pos   = (i, j)

    return score_matrix, trace_matrix, max_score, max_pos


def traceback(seq1, seq2, score_matrix, trace_matrix, max_pos, sub_matrix):
    a1, a2, mid = [], [], []
    i, j = max_pos
    sim_count = 0
    DIAG, UP, LEFT = 1, 2, 3

    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        direction = trace_matrix[i][j]
        if direction == DIAG:
            b1, b2 = seq1[i-1], seq2[j-1]
            a1.append(b1); a2.append(b2)
            is_similar = score(b1, b2, sub_matrix) > 0
            if is_similar:
                sim_count += 1
            mid.append("|" if b1 == b2 else ".")
            i -= 1; j -= 1
        elif direction == UP:
            a1.append(seq1[i-1]); a2.append("-"); mid.append(" ")
            i -= 1
        elif direction == LEFT:
            a1.append("-"); a2.append(seq2[j-1]); mid.append(" ")
            j -= 1
        else:
            break

    a1  = "".join(reversed(a1))
    a2  = "".join(reversed(a2))
    mid = "".join(reversed(mid))
    return a1, a2, mid, sim_count, i, j


def format_results(a1, a2, mid, sim_count, start_i, start_j, max_score, matrix_name):
    lines = []
    length = len(mid)
    identities = mid.count("|")
    gaps = a1.count("-") + a2.count("-")

    lines.append(f"\nLOCAL ALIGNMENT ({matrix_name})")
    lines.append(f"Gap open:   {GAP_OPEN}  |  Gap extend: {GAP_EXTEND}")
    lines.append(f"Max score:  {max_score}")
    lines.append(f"Length:      {length}")
    lines.append(f"Identity:   {identities}/{length} ({100 * identities / length:.1f}%)")
    lines.append(f"Simularity: {sim_count}/{length} ({100 * sim_count / length:.1f}%)")
    lines.append(f"Gaps:       {gaps}/{length} ({100 * gaps / length:.1f}%)\n")

    WIDTH = 60
    pos1 = start_i + 1
    pos2 = start_j + 1
    for s in range(0, length, WIDTH):
        sl = slice(s, s + WIDTH)
        chunk_a1 = a1[sl]
        chunk_mid = mid[sl]
        chunk_a2 = a2[sl]
        lines.append(f"Seq1  {pos1:>4}  {chunk_a1}")
        lines.append(f"      {'':>4}  {chunk_mid}")
        lines.append(f"Seq2  {pos2:>4}  {chunk_a2}\n")
        pos1 += len(chunk_a1) - chunk_a1.count("-")
        pos2 += len(chunk_a2) - chunk_a2.count("-")
    return "\n".join(lines)


def save_results(a1, a2, mid, sim_count, start_i, start_j, max_score, matrix_name, mode):
    if mode == "1":
        filename = "output_DNA.txt"
    else:
        filename = f"output_Protein_{matrix_name}.txt"
    with open(filename, "w") as f:
        f.write(format_results(a1, a2, mid, sim_count, start_i, start_j, max_score, matrix_name))
    print(f"Vysledok ulozeny do {filename}")


def main():
    mode, mat_choice = ask_user()
    sub_matrix, matrix_name = get_matrix(mode, mat_choice)

    seq1 = read_fasta(FILE1)
    seq2 = read_fasta(FILE2)

    if not seq1 or not seq2:
        print("Error: Sequence is empty or missing file.")
        return


    score_matrix, trace_matrix, max_score, max_pos = smith_waterman(seq1, seq2, sub_matrix)
    a1, a2, mid, sim_count, end_i, end_j = traceback(seq1, seq2, score_matrix, trace_matrix, max_pos, sub_matrix)
    save_results(a1, a2, mid, sim_count, end_i, end_j, max_score, matrix_name, mode)


main()