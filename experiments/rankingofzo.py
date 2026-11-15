def map_attributes(mu_diff_file, cosine_rank_file):
    # Lees de mu-difference attributen in volgorde
    mu_attributes = []
    with open(mu_diff_file, 'r') as f:
        for line in f:
            if 'Attribute:' in line:
                attr = int(line.split('Attribute: ')[1].split(',')[0])
                mu_attributes.append(attr)
    
    # Lees de cosine ranking en maak een lookup dictionary
    cosine_ranking = {}
    with open(cosine_rank_file, 'r') as f:
        for rank, line in enumerate(f, 1):
            if line.startswith('Attribute '):
                attr = int(line.split(' ')[1].split('.')[0])
                cosine_ranking[attr] = rank
    
    # Maak de mapping
    result = {}
    for mu_rank, attr in enumerate(mu_attributes, 1):
        cosine_rank = cosine_ranking.get(attr, None)
        if cosine_rank:
            result[mu_rank] = cosine_rank
    
    return result

# Gebruiksvoorbeeld:
mapping = map_attributes('gaussian_rankings/mu_differences_100percent.txt', 'rankingvancub.txt')
for mu_rank, cosine_rank in mapping.items():
    print(f"Mu Difference Rank {mu_rank} → Cosine Rank {cosine_rank}")

