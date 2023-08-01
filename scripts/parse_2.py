with open("..\\transcripts_text\\infosys\\sep-22\\sep-22-1.txt", "r", encoding="utf-8") as fh:
    te = fh.read()

te_sp = te.split("Rishi Basu")
i = 0
for t in te_sp :
    i = i+1
    with open("..\\transcripts_text\\infosys\\sep-22\\sep-22-1-"+str(i)+".txt", "w", encoding="utf-8") as fh:
        fh.write(t)