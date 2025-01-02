# Other Cohesion Papers

## (Review) Coupling and Cohesion Metrics for Object-Oriented Software: A Systematic Mapping Study (2018)



## (C3, LCSM) The Conceptual Cohesion of Classes (2005)

https://www.cs.wm.edu/~denys/pubs/marcusa-Cohesion.pdf

Calculate cohesion scores for all classes of a single project (WinMerge) and look at their pairwise correlations. Finds interesting examples and comments on them.

Compares with LCOM1, LCOM2, LCOM3, LCOM4, LCOM5, ICH, TCC, LCC, Coh, 

>Henderson-Sellers [22] noted that: "It is after all possible to have a class with high internal, syntactic cohesion but little semantic cohesion".

## A suite of metrics for quantifying historical changes to predict future change-prone classes in object-oriented software (2013)

## (LCC) Concern-based cohesion as change proneness indicator: an initial empirical study (2011)


# Finding Consensus

1. Don't distinguish between users. All responses go in the Bradley-Terry Model.

2. Only include "A > B" when *everyone* agrees A > B. Consider every pair.

3. Borda Count

4. Kemeny-Young Method

5. Rank Aggregation via Median Rank

# Process

Two difficulties:

1. There will be a lot of ties. While it is clear when one file is much more cohesive than another, the opposite is not so when asking subjective/qualitative questions.

2. Experts may disagree.

------

1. Ask each user a series of questions. Each question gives the user two files. They may respond, "File A is more cohesive.", "File B is more cohesive." or "They have about the same level of cohesion." Do not ask the same question more than once. Use Thompson sampling to select which question to ask next. Use power analysis to determine how many questions we must ask to get a statistically significant result.

2. Use the Bradley-Terry model to derive a list of cohesion scores for each user.

3. Find a consensus.
- Use a single Bradley-Terry model
- 