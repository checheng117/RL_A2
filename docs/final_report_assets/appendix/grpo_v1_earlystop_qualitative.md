# GRPO-V1 early (700) vs final — qualitative (aligned test indices)

Scoring: reward **v1**; ROUGE-L vs `summary_en_chosen`.

## Example 1 — row index `39`
- **Early strict** / **Final strict**: False / False
- **Early reward** / **Final reward**: 1.0000 / 1.0000 · **Early ROUGE-L** / **Final ROUGE-L**: 0.1505 / 0.1545
- **Reference (`summary_en_chosen`)**:

```
[point]The text explores the historical and rhetorical significance of three sentences from "The Secret History of the Mongols."  
[reason]1. It discusses the implied authority of the Yellow Emperor over the Dada people. 2. It highlights Genghis Khan's assertion of control and loyalty from his subjects. 3. It describes a moment of military and emotional response to a loss of control.  
[summary]These sentences reflect themes of leadership, power, and the dynamics of Mongol society.
```
- **Early prediction (700)**:

```
[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor's authority is emphasized as natural and unavoidable. 2. Chinggis Khan asserts his leadership and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the need for military regrouping.  
[summary]The text serves as a powerful narrative that explores the dynamics of imperial rule and military strategy in the context of Mongol history.
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor's authority is emphasized as natural and unavoidable. 2. Chinggis Khan asserts his leadership and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the need for military regrouping.  
[summary]The text serves as a powerful narrative that explores the dynamics of imperial rule and military strategy in the context of Mongol history.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor's authority is emphasized as natural and unavoidable. 2. Chinggis Khan asserts his leadership and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the need for military regrouping.  
[summary]The text serves as a powerful narrative that explores the dynamics of imperial rule and military strategy in the context of Mongol history.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor's authority is emphasized as natural and unavoidable. 2. Chinggis Khan asserts his leadership and demands the people and their descendants remain under his control. 3. The dialogue reflects the
```
- **Final prediction (723)**:

```
[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor is asked rhetorically why he would not govern the people of Dada. 2. Chinggis Khan asserts his authority and demands that the people remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text captures the power dynamics and political ambitions of both the Yellow Emperor and Chinggis Khan.
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor is asked rhetorically why he would not govern the people of Dada. 2. Chinggis Khan asserts his authority and demands that the people remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text captures the power dynamics and political ambitions of both the Yellow Emperor and Chinggis Khan.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor is asked rhetorically why he would not govern the people of Dada. 2. Chinggis Khan asserts his authority and demands that the people remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text captures the power dynamics and political ambitions of both the Yellow Emperor and Chinggis Khan.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's determination to maintain Mongol control.  
[reason]1. The Yellow Emperor is asked rhetorically why he would not govern the people of Dada. 2. Chinggis Khan asserts his authority and demands that the people remain under his control. 3. The dialogue reflects the tension between imperial authority and the
```

## Example 2 — row index `11`
- **Early strict** / **Final strict**: True / True
- **Early reward** / **Final reward**: 1.0000 / 1.0000 · **Early ROUGE-L** / **Final ROUGE-L**: 0.4598 / 0.4000
- **Reference (`summary_en_chosen`)**:

```
[point]Both computer hardware and peripherals are important, but they serve different roles in the overall gaming experience.  
[reason]1. Hardware acts as the core tool for performance, while peripherals enhance immersion and enjoyment. 2. A powerful computer is useless if paired with poor-quality peripherals that hinder gameplay. 3. The goal of gaming is to have fun, not just to showcase high-end hardware.  
[summary]A great gaming experience comes from balancing strong hardware with quality peripherals that make the experience enjoyable.
```
- **Early prediction (700)**:

```
[point]Gaming is about the overall experience, not just the hardware.  
[reason]1. The computer's hardware is essential for performance, but peripherals like gamepads and keyboards enhance the immersion and enjoyment. 2. A high-end setup with poor peripherals can be frustrating, while a low-end setup with good peripherals can be more enjoyable. 3. The goal of gaming is to have fun, not just to maximize performance.  
[summary]Ultimately, the best gaming experience comes from a combination of hardware and peripherals that together create a fun and immersive experience.
```
- **Final prediction (723)**:

```
[point]Gaming is about the overall experience, not just the hardware.  
[reason]1. The computer's hardware is essential for performance, but peripherals like gamepads and keyboards enhance the experience. 2. A poor keyboard or mouse can make gaming frustrating and unenjoyable. 3. The goal of gaming is to have fun, not just to maximize performance.  
[summary]A good combination of hardware and peripherals is key to a truly enjoyable gaming experience.
```

## Example 3 — row index `76`
- **Early strict** / **Final strict**: True / True
- **Early reward** / **Final reward**: 1.0000 / 1.0000 · **Early ROUGE-L** / **Final ROUGE-L**: 0.3818 / 0.4080
- **Reference (`summary_en_chosen`)**:

```
[point]China's purchase of 292 Airbus planes is a strategic move to complement its homegrown C919 and maintain a competitive, diversified aviation fleet.  
[reason]1. The C919 is still new and not yet as reliable or proven as established models like the Airbus A320. 2. Airbus offers a wider range of aircraft types, including wide-body planes, which are essential for diverse route and cargo needs. 3. The order strengthens China's bargaining power in the global aviation market and reinforces diplomatic ties with Europe.  
[summary]China's decision reflects a balance between supporting domestic industry and ensuring global competitiveness and cooperation.
```
- **Early prediction (700)**:

```
[point]China is still purchasing Airbus planes despite having its own C919 aircraft because the C919 is a new, limited, and single-aisle aircraft that still needs to be a reliable workhorse.  
[reason]1. The C919 is a relatively new and limited aircraft type that cannot yet meet the demand for a diverse and reliable fleet. 2. Airbus offers a wide range of aircraft types, including wide-body planes, to cater to different routes and passenger needs. 3. The Airbus order is a strategic move to maintain competitiveness, secure global deals, and strengthen international trade relations.  
[summary]China's Airbus purchase is a calculated decision to ensure its airlines remain competitive and connected to the global market.
```
- **Final prediction (723)**:

```
[point]China is still purchasing Airbus planes despite having its own C919 aircraft because the C919 is a new and limited aircraft type.  
[reason]1. The C919 is a relatively new and limited aircraft, not yet a proven workhorse. 2. Airbus offers a wide range of aircraft types, including wide-body planes, to meet diverse airline needs. 3. China is a pragmatic global player that benefits from large Airbus orders to maintain competitiveness and international trade relations.  
[summary]China's Airbus purchase is a strategic move to ensure its airlines remain competitive and connected to the global market.
```
