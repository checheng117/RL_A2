# GRPO-V1 E2-dense — early vs final qualitative

Early: **grpo_v1_e2_dense_step200** · Final: **grpo_v1_e2_dense_step300**

## Example 1 (row `75`)
- strict early/final: True / True
- reward early/final: 1.0000 / 1.0000 · ROUGE-L: 0.3842 / 0.3350
- reference:
```
[point]The academic community is hesitant to recognize the Xia Dynasty due to a lack of concrete evidence.  
[reason]1. The Xia Dynasty lacks unified and comprehensive historical records. 2. Interpretations of ancient texts, such as oracle bones, remain ambiguous. 3. Titles like "后" (hòu) used for Xia rulers are not sufficient to confirm the dynasty's existence on their own.  
[summary]Ongoing archaeological discoveries may one day provide the clarity needed to validate the Xia Dynasty's place in history.
```
- early:
```
[point]The academic community hesitates to recognize the Xia Dynasty due to a lack of clear and consistent evidence.  
[reason]1. The use of the term "后" (hòu) for Xia rulers is consistent and unlikely to be a personal name. 2. The Xia Dynasty is not well-documented in the oracle bones, making it difficult to verify its existence. 3. The complexity of interpreting ancient texts has led to ongoing debates and uncertainty.  
[summary]Despite these challenges, continued research and new evidence may eventually lead to the recognition of the Xia Dynasty as a historical period.
```
- final:
```
[point]The academic community hesitates to recognize the Xia Dynasty due to a lack of clear, unified evidence and the complexity of interpreting ancient texts.  
[reason]1. The use of the character "后" (hòu) for rulers during the Xia Dynasty is consistent and unlikely to be a mistake. 2. The consistent reference to Xia rulers as "后" (hòu) across different periods, including before the Tang Dynasty, supports the idea of a legitimate historical period. 3. The scarcity of concrete evidence and ongoing debates among historians have hindered the recognition of the Xia Dynasty.  
[summary]As new archaeological and textual findings emerge, the possibility of recognizing the Xia Dynasty as a legitimate historical period becomes increasingly plausible.
```

## Example 2 (row `39`)
- strict early/final: True / True
- reward early/final: 1.0000 / 1.0000 · ROUGE-L: 0.3218 / 0.3046
- reference:
```
[point]The text explores the historical and rhetorical significance of three sentences from "The Secret History of the Mongols."  
[reason]1. It discusses the implied authority of the Yellow Emperor over the Dada people. 2. It highlights Genghis Khan's assertion of control and loyalty from his subjects. 3. It describes a moment of military and emotional response to a loss of control.  
[summary]These sentences reflect themes of leadership, power, and the dynamics of Mongol society.
```
- early:
```
[point]The text is a dramatic and rhetorical exploration of the Yellow Emperor's authority and Chinggis Khan's assertion of control over the Mongol people.  
[reason]1. The Yellow Emperor is questioned about his ability to govern the people of Dada. 2. Chinggis Khan claims to have taken the people and their descendants as his vassals. 3. A lament is made about the loss of people and the need for military leaders to be recalled.  
[summary]The passage blends historical references with rhetorical flair to highlight the power dynamics between the Yellow Emperor and Chinggis Khan.
```
- final:
```
[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor's authority is emphasized as natural and inevitable, as he is the legendary ruler who would govern the people. 2. Chinggis Khan asserts his authority and challenges the Yellow Emperor's rule, promising to keep the people under his control. 3. The dialogue includes a lament about the loss of people and a call for the Mongol troops to regroup.  
[summary]The text uses a dramatic and rhetorical style to explore themes of authority, control, and the inevitability of historical power.
```

## Example 3 (row `53`)
- strict early/final: True / True
- reward early/final: 1.0000 / 1.0000 · ROUGE-L: 0.4512 / 0.4000
- reference:
```
[point]The responder is eager to help describe mythological creatures from "The Founder of Diabolos" that are not part of the main storyline.  
[reason]1. The responder acknowledges the user's deep engagement with the novel. 2. They clarify that their descriptions are interpretations and invite corrections. 3. They ask for specifics to tailor the descriptions accordingly.  
[summary]The responder aims to provide vivid, customized descriptions of non-main mythological creatures based on the user's preferences.
```
- early:
```
[point]The speaker is interested in describing various mythological creature forms from "The Founder of Diabolos" that are not part of the main storyline.  
[reason]1. The speaker is a wizard of the English language and is eager to help the reader explore the book. 2. They are open to descriptions of humanoid, beastly, or unconventional creature forms. 3. They invite the reader to specify which creatures they would like to see.  
[summary]The speaker is ready to provide vivid descriptions of the book's mythological creatures.
```
- final:
```
[point]The answer invites the reader to explore and describe the mythological creature forms from "The Founder of Diabolos" that are not part of the main storyline.  
[reason]1. The speaker is enthusiastic and eager to help the reader discover more about the book. 2. They mention their expertise in the English language and offer to describe the creatures vividly. 3. They ask the reader to specify which types of creatures they are interested in, such as humanoid or beastly forms.  
[summary]The response is friendly and open to engagement, encouraging the reader to share their interests and receive a descriptive account of the book's mythological elements.
```
