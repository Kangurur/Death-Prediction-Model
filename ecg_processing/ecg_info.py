import pandas as pd
import json

import os
import json
import pandas as pd

folder = 'data/digitized_json_files'

min_leads = 12
min_lenght = float('inf')

for filename in os.listdir(folder):
    if filename.endswith('.json'):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as f:
            raw = json.load(f)
        leads = raw['leads']
        for i in range(len(leads)):
            lead = leads[i]['signal']
            print(f"Lead {i+1} from {filename}:{len(lead)}")
            min_lenght = min(min_lenght, len(lead))
        min_leads = min(min_leads, len(leads))
        
print(f"Minimum number of leads: {min_leads}")
print(f"Minimum length of leads: {min_lenght}")