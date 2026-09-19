/**
 * The landing page's video assets.
 *
 * These are hosted on a third-party CDN rather than in `public/`, which is a
 * departure from the rule the console holds to — self-hosted, no network
 * required. Two consequences are handled in code rather than hoped away:
 *
 *   - Every video sits on top of the CSS dream wall and starts at opacity 0,
 *     revealing itself only once the browser reports it can actually play. A
 *     blocked CDN therefore leaves the schematic visible rather than a black
 *     rectangle, so the page still renders with no network at all.
 *   - Nothing here is model output, and the page says so. The generated frames
 *     that carry provenance live in the console.
 */
const CDN = "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P";

export const VIDEO = {
  hero: `${CDN}/hf_20260405_074625_a81f018a-956b-43fb-9aee-4d1508e30e6a.mp4`,
  calledShot: `${CDN}/hf_20260402_054547_9875cfc5-155a-4229-8ec8-b7ba7125cbf8.mp4`,
  pipeline: `${CDN}/hf_20260307_083826_e938b29f-a43a-41ec-a153-3d4730578ab8.mp4`,
  research: `${CDN}/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4`,
  craft: `${CDN}/hf_20260324_151826_c7218672-6e92-402c-9e45-f1e0f454bdc4.mp4`,
} as const;
