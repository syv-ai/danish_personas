Du foreslår kun små rettelser til en allerede skrevet dansk persona. Den eksisterende
tekst skal forblive næsten uændret. Brug kun de oplyste ændrede fakta som grundlag;
opfind ikke nye forhold, og udled ikke personlige egenskaber af køn eller partnerkøn.

Returnér udelukkende et JSON-objekt med feltet `patches`. Hver patch har felterne
`old_excerpt` og `new_excerpt`. Kopiér `old_excerpt` tegnret fra den oprindelige
persona; det skal optræde præcis én gang. Foreslå højst to korte udskiftninger, der
kun retter konkrete modsætninger til de nye fakta. Skriv ikke personaen om, og ret
ikke andre sætninger af stilistiske grunde. Bevar navne, detaljer og tone uden for
udskiftningerne. Hvert udsnit skal være højst 120 tegn, og samlet skal rettelserne
være væsentligt mindre end originalen.

Hvis rettelsen er tvetydig, kræver flere eller større ændringer, eller ingen sikker
lokal udskiftning findes, så returnér `{"patches": []}`. Det er bedre at undlade en
rettelse end at ændre betydning eller tilføje nye påstande.
