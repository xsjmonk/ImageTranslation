import csv, os
class CsvPredictionWriter:
    fields=("image_id","image_path","label","taken_probability","not_taken_probability","requires_review","model_version","status","error")
    def __init__(self,path): self.path=path
    def write(self,result):
        tmp=str(self.path)+".tmp"
        with open(tmp,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=self.fields); w.writeheader()
            for p in result.predictions: w.writerow({"image_id":p.image_id,"image_path":p.image_path,"label":p.label.value,"taken_probability":p.taken_probability,"not_taken_probability":p.not_taken_probability,"requires_review":p.requires_review,"model_version":p.model_version,"status":"ok","error":""})
            for x in result.failures: w.writerow({"image_id":x.image_id,"image_path":x.image_path,"status":"failed","error":x.error})
        os.replace(tmp, self.path)
