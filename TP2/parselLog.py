import csv
import re

input_file = "log.txt"
output_file = "logParsed.csv"

# regex para capturar sec y nanosec
time_pattern = re.compile(r"sec=(\d+), nanosec=(\d+)")

rows = []

with open(input_file, "r") as infile:
    for line in infile:
        time_match = time_pattern.search(line)
        if not time_match:
            continue

        sec, nanosec = time_match.groups()
        # generar timestamp como sec.nanosec
        timestamp = f"{sec}{nanosec}"

        # extraer los valores flotantes después del tabulador
        parts = line.split("\t")[1:]
        floats = [p.strip() for p in parts if p.strip()]

        if(float(floats[1]) > 0.001 and float(floats[2]) > 0.001):
            started = True

        # fila con timestamp + valores
        if(started):
            row = [timestamp] + floats
            rows.append(row)

        

# armar encabezados dinámicos según cantidad de columnas
if rows:
    num_cols = len(rows[0]) - 1
    header = ["timestamp"] + [f"col{i+1}" for i in range(num_cols)]

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

print(f"CSV generado: {output_file}")
