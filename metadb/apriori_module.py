def apriori():
    print("""
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the groceries dataset and preprocess it to get transaction lists
df_raw = pd.read_csv("Groceries_dataset.csv")

# Combine Member_number and Date to form a transaction key
df_raw["Transaction"] = df_raw["Member_number"].astype(str) + "_" + df_raw["Date"]
transactions = df_raw.groupby("Transaction")["itemDescription"].apply(list).tolist()

# Step 2: Use the Apriori algorithm to generate frequent itemsets with min_support=0.01
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Step 3: Generate association rules with min_confidence=0.3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# Step 4: Sort the generated rules by lift and display the top 5
top_rules = rules.sort_values(by="lift", ascending=False).head(5)
print("Top 5 Association Rules by Lift:")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 5: Visualize item frequency using a bar chart
item_freq = df.sum().sort_values(ascending=False).head(10)
item_freq.plot(kind='bar', color='skyblue')
plt.title("Top 10 Most Frequent Items")
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 6: Analyze and explain one strong rule
best_rule = top_rules.iloc[0]
print("\\nStrong Rule Analysis:")
print(f"If a customer buys {set(best_rule['antecedents'])}, they are likely to buy {set(best_rule['consequents'])}.")
print(f"Support: {best_rule['support']:.2f}, Confidence: {best_rule['confidence']:.2f}, Lift: {best_rule['lift']:.2f}")       
          """)