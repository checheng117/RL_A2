# DPO retune v2 qualitative analysis

Indices from qualitative_pick_indices.json: [16, 30, 61, 25, 39].
Models: SFT, DPO retune v1, DPO retune v2 (same greedy decoding).

## Example 1 (row index 16)
- **Example ID**: 1 (index 16)
- **Reference summary**:
```
[point]The question is based on a mix-up of characters from different Chinese literary works.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms," and they do not appear in the same story. 2. The Six-Eared Macaque is also from "Journey to the West," but Sun Wuzhong is not a character in that novel.  
[summary]The confusion arises from combining characters from separate stories, which leads to an inaccurate question.
```
### SFT prediction
```
[point]The question is based on a mix-up of characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but it is not trying to escape from Sun Wuzhong, who is not a character in the same story.  
[summary]The confusion stems from mixing characters from different works.
```
- strict_format: **Pass**; ROUGE-L: **0.7468**
### DPO retune v1
```
[point]The question is based on a mix-up of characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but it is not trying to escape from Sun Wuzhong, who is not a character in the same story.  
[summary]The confusion stems from mixing characters from different works.
```
- strict_format: **Pass**; ROUGE-L: **0.7468**
### DPO retune v2
```
[point]The question is based on a mix-up of characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but it is not trying to escape from Sun Wuzhong, who is not a character in the same story.  
[summary]The confusion stems from mixing characters from different works.
```
- strict_format: **Pass**; ROUGE-L: **0.7468**
- **Notes**: Similar ROUGE-L v1 vs v2.

## Example 2 (row index 30)
- **Example ID**: 2 (index 30)
- **Reference summary**:
```
[point]Modernist and Postmodernist design differ in their approaches and aesthetics but share a common concern for the role of design in society.  
[reason]1. Modernist design emphasizes functionality, simplicity, and universality, while Postmodernist design values playfulness, eclecticism, and contextual relevance. 2. Postmodernist design emerged as a reaction to the rigidity of Modernist design, challenging its norms and reinterpreting its elements. 3. Both movements reflect on the relationship between design and society, albeit from different perspectives.  
[summary]Despite their differences, Modernist and Postmodernist design both contribute to the evolving dialogue on the purpose and impact of design.
```
### SFT prediction
```
[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude, but both share a common concern for the role of design in society.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness, eclecticism, and contextual relevance. 2. Postmodernist design emerged as a reaction to the limitations of Modernist design, building upon its foundational principles. 3. Both movements reflect a shared interest in how design shapes and reflects human society.  
[summary]While distinct in their approaches, Modernist and Postmodernist design both contribute to the evolution of design as a tool for societal influence.
```
- strict_format: **Pass**; ROUGE-L: **0.7041**
### DPO retune v1
```
[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude, though they share a common concern for design's role in society.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness, eclecticism, and contextual relevance. 2. Postmodernist design reacted against the strict rules of Modernist design, blending and reinterpreting its elements. 3. Both movements explored the relationship between design and society, though from distinct perspectives.  
[summary]While distinct, Modernist and Postmodernist design both reflect the evolving role of design in shaping our world.
```
- strict_format: **Pass**; ROUGE-L: **0.6952**
### DPO retune v2
```
[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude, though they share a common concern for design's role in society.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness, eclecticism, and contextual relevance. 2. Postmodernist design emerged as a reaction to the constraints of Modernist design, building upon its foundational principles. 3. Both movements reflect a shared interest in how design shapes and reflects human society.  
[summary]While distinct, Modernist and Postmodernist design both contribute to the evolution of design as a tool for societal influence.
```
- strict_format: **Pass**; ROUGE-L: **0.6667**
- **Notes**: v1 higher ROUGE-L than v2.

## Example 3 (row index 61)
- **Example ID**: 3 (index 61)
- **Reference summary**:
```
[point]Cao Cao's pragmatic approach to absorbing the Yellow Turban rebels was more effective than Liu Bei's rigid and people-centric strategy.  
[reason]1. Cao Cao recognized the value of former rebels and integrated them into his army, showcasing adaptability. 2. Liu Bei, despite claiming to care for the people, fought against the very peasants he supposedly supported. 3. Cao Cao's decision demonstrated superior crisis management and long-term strategic thinking.  
[summary]In the turbulent Three Kingdoms era, Cao Cao's flexibility and willingness to evolve ultimately gave him an edge over Liu Bei's inflexible ideals.
```
### SFT prediction
```
[point]Cao Cao's approach of absorbing the Yellow Turban rebels into his army was a strategic and effective decision.  
[reason]1. Cao Cao recognized the value of the Yellow Turban rebels and offered them employment, integrating them into his army. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately failed to suppress the Yellow Turban rebels effectively. 3. Cao Cao's pragmatism allowed him to adapt to the changing situation, while Liu Bei's rigid ideology led to failure.  
[summary]Cao Cao's ability to adapt and capitalize on the situation ultimately proved more effective in the Three Kingdoms era.
```
- strict_format: **Pass**; ROUGE-L: **0.4138**
### DPO retune v1
```
[point]Cao Cao's approach of absorbing the Yellow Turban rebels was more effective than Liu Bei's.  
[reason]1. Cao Cao recognized the value of the Yellow Turban rebels and offered them employment, integrating them into his army. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately had to fight against the very people he claimed to serve. 3. Cao Cao's strategy demonstrated his ability to adapt and capitalize on the crisis, while Liu Bei's actions were seen as desperate and ineffective.  
[summary]Cao Cao's pragmatic and adaptive approach ultimately proved more successful in the Three Kingdoms era.
```
- strict_format: **Pass**; ROUGE-L: **0.5196**
### DPO retune v2
```
[point]Cao Cao's approach of absorbing the Yellow Turban rebels into his army was more effective and strategic than Liu Bei's.  
[reason]1. Cao Cao recognized the value of the Yellow Turban rebels and offered them employment, integrating them into his army. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately had to fight against the very people he claimed to care about. 3. Cao Cao's pragmatism and adaptability contrasted with Liu Bei's rigid adherence to ideology.  
[summary]Cao Cao's clever and flexible strategy ultimately proved more effective in the complex political landscape of the Three Kingdoms.
```
- strict_format: **Pass**; ROUGE-L: **0.5000**
- **Notes**: Similar ROUGE-L v1 vs v2.

## Example 4 (row index 25)
- **Example ID**: 4 (index 25)
- **Reference summary**:
```
[point]Tsunade left Konoha due to disillusionment and the village chose not to call her back to respect her decision and allow her time to heal.  
[reason]1. Tsunade was disillusioned by the village's politics and the constant wars, which conflicted with her ideals of peace and protection. 2. The Third Hokage, Hiruzen, respected her choice to leave and understood that forcing her return would cause more pain. 3. Hiruzen believed Tsunade would return when ready, as she was needed for Konoha's future.  
[summary]Tsunade's departure and eventual return reflect a journey of personal growth and the wisdom of the Third Hokage in trusting her path.
```
### SFT prediction
```
[point]Tsunade left Konoha because she was disillusioned with the village's politics and the constant wars, and Hiruzen Sarutobi chose not to call her back to respect her and her journey.  
[reason]1. Tsunade felt the village was stagnant and her ideals of peace were being compromised. 2. She needed time to heal and find herself after the death of her loved one, Dan Katō. 3. Hiruzen respected her decision and understood the need for a strong, compassionate leader to lead the village.  
[summary]Tsunade's departure and return were a pivotal moment in Konoha's history, shaped by her personal growth and the village's need for a wise and just leader.
```
- strict_format: **Pass**; ROUGE-L: **0.4126**
### DPO retune v1
```
[point]Tsunade left Konoha because she was disillusioned with the village's politics and needed time to heal, and Hiruzen Sarutobi chose not to call her back to respect her and ensure a better future.  
[reason]1. Tsunade felt the village was stagnant and her ideals of peace were being compromised. 2. She had lost a loved one and realized she could not serve a village that didn't align with her values. 3. Hiruzen respected her decision, understanding that she needed time to recover and that she would return when the right moment came.  
[summary]Tsunade's departure and Konoha's inaction were reflections of self-discovery, trust, and the need for a stronger, more just leadership.
```
- strict_format: **Pass**; ROUGE-L: **0.3877**
### DPO retune v2
```
[point]Tsunade left Konoha because she was disillusioned with the village's politics and the constant wars, and Hiruzen Sarutobi chose not to call her back to respect her and her journey.  
[reason]1. Tsunade felt the village was stagnant and her ideals of peace were being compromised. 2. She needed time to heal and find herself after the death of her loved one, Dan Katō. 3. Hiruzen respected her decision and believed she would return when the right time came.  
[summary]Tsunade's departure and return were a pivotal moment in Konoha's history, reflecting the complexities of leadership and self-discovery.
```
- strict_format: **Pass**; ROUGE-L: **0.4340**
- **Notes**: v2 higher ROUGE-L than v1. v2 shorter.

## Example 5 (row index 39)
- **Example ID**: 5 (index 39)
- **Reference summary**:
```
[point]The text explores the historical and rhetorical significance of three sentences from "The Secret History of the Mongols."  
[reason]1. It discusses the implied authority of the Yellow Emperor over the Dada people. 2. It highlights Genghis Khan's assertion of control and loyalty from his subjects. 3. It describes a moment of military and emotional response to a loss of control.  
[summary]These sentences reflect themes of leadership, power, and the dynamics of Mongol society.
```
### SFT prediction
```
[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the power dynamics between the Yellow Emperor and Chinggis Khan.
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the power dynamics between the Yellow Emperor and Chinggis Khan.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain...
```
- strict_format: **Fail**; ROUGE-L: **0.1509**
### DPO retune v1
```
[point]The text presents a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor is questioned by the text, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands the people remain under his control, framing it as a matter of loyalty and necessity. 3. The dialogue includes a lament over the loss of people, emphasizing the urgency and conflict between the two leaders.  
[summary]The text uses a dramatic and rhetorical style to explore the tension between imperial authority and the demands of loyalty and survival.
[point]The text presents a dramatic dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor is questioned, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands the people remain under his control, framing it as a matter of loyalty and necessity. 3. The dialogue includes a lament over the loss of people, emphasizing the urgency and conflict between the two leaders.  
[summary]The text uses a dramatic and rhetorical style to explore the tension between imperial authority and the demands of loyalty and survival.
[point]The text presents a dramatic dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor is questioned, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands t...
```
- strict_format: **Fail**; ROUGE-L: **0.1670**
### DPO retune v2
```
[point]The text presents a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor is questioned by the text, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue includes a lament about the loss of people and a call for the return of military leaders.  
[summary]The text uses a dramatic and rhetorical style to explore themes of authority, control, and the inevitability of historical power.
assistant[point]The text presents a dramatic dialogue between the Yellow Emperor and Chinggis Khan, exploring themes of authority and control.  
[reason]1. The Yellow Emperor is questioned, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. The dialogue includes a lament about the loss of people and a call for military leaders to return.  
[summary]The text uses a dramatic and rhetorical style to explore the inevitability of historical power and authority.
assistant[point]The text presents a dramatic dialogue between the Yellow Emperor and Chinggis Khan, exploring themes of authority and control.  
[reason]1. The Yellow Emperor is questioned, suggesting it is natural for him to govern the people of Dada. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. The dialogue includes a lament about the loss of people and a call for military leaders to return.  
[summary]The text uses a dramatic and rhetorical style to explore the inevitability of historical power ...
```
- strict_format: **Fail**; ROUGE-L: **0.1674**
- **Notes**: Similar ROUGE-L v1 vs v2.

## Summary

### Retune v2 vs v1
See aggregate `dpo_retune_v2_test_metrics.json` and four-way table (`outputs/report_assets/sft_vs_dpo_all_metrics.md`).

### Aggregate vs per-example
On the **full English test split (n=81)**, **DPO retune v2** is **at or slightly above SFT** on ROUGE-L and strict format, with **shorter** mean length — see `outputs/report_assets/dpo_final_decision.md`. Per-example ROUGE can still favor v1 on some rows (e.g. index 30); report both **aggregate** and **qualitative** without claiming uniform dominance.

### Part II final pick
**DPO retune v2** is the submission-facing DPO; document **original DPO failure** + **retune path** honestly.
