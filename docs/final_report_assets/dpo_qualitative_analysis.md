# DPO vs SFT qualitative comparison

Same **test row indices** as `sft_qualitative_analysis.md`: `[16, 30, 61, 25, 39]`.

## Example 1 (index `16`)
- **Reference (`summary_en_chosen`)**:
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
- ROUGE-L: **0.7468**; format: **Pass**
### DPO prediction
```
[point]The question is based on a confusion between characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but he is not trying to escape from Sun Wuzhong, who is not a character in that story.  
[summary]The confusion stems from mixing characters from unrelated works.
[point]The question is based on a confusion between characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but he is not trying to escape from Sun Wuzhong, who is not a character in that story.  
[summary]The confusion arises from mixing characters from unrelated works.
[point]The question is based on a confusion between characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is from "Journey to the West," but he is not trying to escape from Sun Wuzhong, who is not a character in that story.  
[summary]The confusion stems from mixing characters from unrelated works.
[point]The question is based on a confusion between characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is from "Journey to the West," but he is not trying to escape from Sun Wuzhong, who is not a character in that story.  
[summary]The confusion arises from mixing characters from unrelated works.
[point]The question is based on a confusion between characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is from "Journey to the West," but he is not trying to escape from Sun W
```
- ROUGE-L: **0.2998**; format: **Pass**
### Comparison
- **Structure**: Both structured.
- **Faithfulness (proxy)**: SFT closer to reference by ROUGE-L.
- **Fluency / length**: DPO output is longer; check for verbosity.

## Example 2 (index `30`)
- **Reference (`summary_en_chosen`)**:
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
- ROUGE-L: **0.7041**; format: **Pass**
### DPO prediction
```
[point]Modernist and Postmodernist design differ primarily in their focus, aesthetic, and attitude toward design.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness, eclecticism, and contextual relevance. 2. Modernist design is rational and structured, whereas Postmodernist design is ironic and challenging traditional norms. 3. Both movements build upon each other, with Postmodernist design often reinterpreting and adapting elements from Modernist design.  
[summary]While distinct, Modernist and Postmodernist design share a common commitment to how design shapes society.
[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude toward design.  
[reason]1. Modernist design emphasizes functionality, simplicity, and universality, while Postmodernist design is more playful, eclectic, and contextual. 2. Modernist design is rational and structured, whereas Postmodernist design challenges traditional norms and embraces irony. 3. Both movements build upon each other, with Postmodernist design often reinterpreting elements from Modernist design.  
[summary]Despite their differences, both Modernist and Postmodernist design reflect a shared commitment to how design influences society.[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude toward design.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness and contextual relevance. 2. Modernist design is rational and structured, whereas Postmodernist design challenges traditional norms and irony. 3. Both movements build upon each other, with Postmodernist design often reinterpreting elements from Modernist design.  
[summary]Both Modernist and Postmodernist design reflect a shared commitment to how design shapes society.[point]Modernist and Postmodernist design differ in their focus, aesthetics, and 
```
- ROUGE-L: **0.2488**; format: **Pass**
### Comparison
- **Structure**: Both structured.
- **Faithfulness (proxy)**: SFT closer to reference by ROUGE-L.
- **Fluency / length**: DPO output is longer; check for verbosity.

## Example 3 (index `61`)
- **Reference (`summary_en_chosen`)**:
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
- ROUGE-L: **0.4138**; format: **Pass**
### DPO prediction
```
[point]Cao Cao's approach of absorbing the Yellow Turban rebels was more effective than Liu Bei's.  
[reason]1. Cao Cao recognized the value of the Yellow Turban rebels and offered them employment, integrating them into his army. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately had to fight against the very people he claimed to serve. 3. Cao Cao's strategy demonstrated strategic adaptability, while Liu Bei's actions reflected a more desperate and rigid approach.  
[summary]Cao Cao's pragmatic and flexible strategy ultimately proved more successful in the Three Kingdoms era.
[point]Cao Cao's strategy of integrating the Yellow Turban rebels was more effective than Liu Bei's.  
[reason]1. Cao Cao recognized the value of the rebels and offered them employment, turning them into allies. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately had to fight against the very people he claimed to serve. 3. Cao Cao's approach showcased strategic adaptability, while Liu Bei's actions reflected a more rigid and desperate approach.  
[summary]Cao Cao's pragmatic and flexible strategy ultimately proved more successful in the Three Kingdoms era.
```
- ROUGE-L: **0.4096**; format: **Pass**
### Comparison
- **Structure**: Both structured.
- **Faithfulness (proxy)**: Roughly comparable faithfulness (ROUGE-L).
- **Fluency / length**: DPO output is longer; check for verbosity.

## Example 4 (index `25`)
- **Reference (`summary_en_chosen`)**:
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
- ROUGE-L: **0.4126**; format: **Pass**
### DPO prediction
```
[point]Tsunade left Konoha because she was disillusioned with the village's politics and needed time to heal, and the Third Hokage chose not to call her back to respect her and ensure a better future.  
[reason]1. Tsunade felt the village was stagnant and her ideals of peace were being compromised. 2. She had lost a loved one and realized she could not serve a village that didn't align with her values. 3. Hiruzen respected her decision, understanding that she needed time to recover and that she would return when the right moment came.  
[summary]Tsunade's departure and the village's inaction were a result of her personal growth and the Third Hokage's wisdom in supporting her journey.
[point]Tsunade left Konoha because she was disillusioned with the village's politics and needed time to heal.  
[reason]1. She felt the village was stagnant and her ideals of peace were being compromised. 2. She had lost a loved one and realized she could not serve a village that didn't align with her values. 3. Hiruzen respected her decision, understanding that she needed time to recover and would return when the right moment came.  
[summary]Tsunade's departure and the village's inaction were a testament to her growth and the Third Hokage's wisdom in supporting her journey.
```
- ROUGE-L: **0.3313**; format: **Pass**
### Comparison
- **Structure**: Both structured.
- **Faithfulness (proxy)**: SFT closer to reference by ROUGE-L.
- **Fluency / length**: DPO output is longer; check for verbosity.

## Example 5 (index `39`)
- **Reference (`summary_en_chosen`)**:
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
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the powe
```
- ROUGE-L: **0.1509**; format: **Pass**
### DPO prediction
```
[point]The text presents a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of governance and the struggle for control.  
[reason]1. The Yellow Emperor questions why the people of Dada would not be under his rule. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. A lament is made about the loss of people and the need for military regrouping.  
[summary]The passage captures the tension between authority, loyalty, and the challenges of governance.
[point]The text presents a dramatic dialogue between the Yellow Emperor and Chinggis Khan, emphasizing the inevitability of governance and the struggle for control.  
[reason]1. The Yellow Emperor questions why the people of Dada would not be under his rule. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. A lament is made about the loss of people and the need for military regrouping.  
[summary]The passage captures the tension between authority, loyalty, and the challenges of governance.
[point]The text highlights the inevitability of governance and the struggle for control between the Yellow Emperor and Chinggis Khan.  
[reason]1. The Yellow Emperor questions why the people of Dada would not be under his rule. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. A lament is made about the loss of people and the need for military regrouping.  
[summary]The passage captures the tension between authority, loyalty, and the challenges of governance.
[point]The text portrays a dramatic confrontation between the Yellow Emperor and Chinggis Khan, emphasizing the inevitability of governance and the struggle for control.  
[reason]1. The Yellow Emperor questions why the people of Dada would not be under his rule. 2. Chinggis Khan asserts his authority and demands the people remain under his control. 3. A lament is made about the loss of people and the ne
```
- ROUGE-L: **0.1735**; format: **Pass**
### Comparison
- **Structure**: Both structured.
- **Faithfulness (proxy)**: DPO closer to reference by ROUGE-L.
- **Fluency / length**: Similar verbosity.

## Summary

### DPO vs SFT — improvements
- DPO may better separate **chosen vs rejected** style when the preference signal aligns with format and concision.

### DPO vs SFT — limitations
- If the rejected summaries are noisy, DPO can **degrade** ROUGE or faithfulness; monitor test metrics.
- LoRA capacity is smaller than full fine-tune; large behavior changes are unlikely without longer training.

### Worth continuing to GRPO?
- If DPO improves **format adherence** or human-judged quality without collapsing ROUGE, GRPO is a reasonable next step for reward shaping experiments.
- If metrics regress, fix preference data or β before spending budget on online RL.
