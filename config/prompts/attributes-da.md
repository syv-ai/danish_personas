# Danske baggrundsattributter

Du skaber strukturerede, syntetiske baggrundsattributter til en dansk persona.
Returnér kun gyldigt JSON efter skemaet. Returnér altid alle skemafelter; brug
null for nullable felter, når værdien ikke findes.

Inputfeltet `job_function` er den officielle jobfunktionsetiket. Feltet
`allowed_job_titles` er den komplette, lukkede liste over tilladte titler for etiketten.
Hvis `allowed_job_titles` er tom, skal `job_title` være null. Ellers skal `job_title`
være præcis én værdi fra listen, inklusive stavning og mellemrum. Brug aldrig koder.
Opfind ikke arbejdsopgaver, arbejdsgiver, institution, anciennitet eller
ledelsesansvar.

Regler:

- Bevar de faste demografiske og OCEAN-input uden at ændre dem.
- Skriv naturligt dansk; jobtitlen skal være trimmet og må ikke indeholde linjeskift.
- Opfind ikke navn, adresse, arbejdsplads, uddannelsesinstitution eller kontaktoplysninger.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke religion, etnicitet, helbred, seksualitet, politisk overbevisning eller
  kriminalhistorik.
- Undgå stereotyper baseret på køn, alder, region, uddannelse eller arbejdsstatus.
- Variér emnerne på tværs af personer; brug ikke automatisk standardkombinationen
  læsning, gåture og madlavning.
- Skriv 3-6 korte, forskellige færdigheder og 3-6 korte, forskellige interesser.
- Brug null til karrieremål, hvis et konkret mål ikke er naturligt ud fra inputtet.
- Brug `current_status` som den nøjagtige aktuelle status, når den er leveret.
- Færdighederne er syntetiske muligheder, ikke dokumenteret erfaring. Undgå konkrete
  arbejdsopgaver, arbejdsgivere, institutioner og tidligere arbejde.
