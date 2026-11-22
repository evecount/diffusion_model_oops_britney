"""
Curated Cinematographic & Music-Video Datasets for Autoregressive-Guided Video Diffusion:
1. GENERATIVE_PARADIGMS: Autoregressive vs. Encoder-Decoder vs. Diffusion vs. Hybrid AR-Diffusion
2. CHURCH_DOOR_CONTINUITY_CASE: The Church Door & Brass Handle Invariant (Hidden AR World-State outside the human prompt)
3. MUSIC_VIDEO_PACKS:
   - britney_oops_i_did_it_again ("Oops!... I Did It Again" - Britney Spears, Dir. Nigel Dick, 2000)
   - britney_baby_one_more_time ("...Baby One More Time" - Britney Spears, Dir. Nigel Dick, 1998)
   - decemberists_o_valencia ("O Valencia!" - The Decemberists, Narrative Folk-Rock Tragedy & Cross-Cut Continuity)
   - michael_jackson_thriller ("Thriller" - Michael Jackson, Dir. John Landis, 1983, Practical Metamorphosis & 3-Layer Reality)
   - michael_jackson_remember_the_time ("Remember the Time" - Michael Jackson, Dir. John Singleton, 1992, Ancient Egypt Sand-Particle Morph Diffusion)
   - miley_cyrus_wrecking_ball ("Wrecking Ball" - Miley Cyrus, Dir. Terry Richardson, 2013, Intimate Tears vs. Kinetic Demolition Contrast)
   - elton_john_your_song ("Your Song" - Elton John, Lyrical Acoustic Intimacy & Continuous Camera Drift)
   - kpop_three_movement_archetype (High-Velocity K-Pop 3-Movement Intercut Benchmark)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class GenerativeParadigmComparison:
    paradigm_id: str
    name: str
    cognitive_intuition: str
    mathematical_form: str
    horse_in_grass_behavior: str
    cinematic_strength: str
    cinematic_blind_spot: str


@dataclass
class WorldStateObject:
    object_id: str
    surface_attributes: Dict[str, str]
    occluded_or_offscreen_state: Dict[str, str]
    persistence_rule: str


@dataclass
class ShotTransitionSpec:
    from_shot_id: str
    to_shot_id: str
    dp_transition_type: str
    duration_sec: float
    audio_cue_anchor: str
    latent_conditioning_bridge: str


@dataclass
class MusicVideoMovementShot:
    shot_id: str
    timecode: str
    movement_layer: str  # "Movement I: Narrative World", "Movement II: Choreography Grid", "Movement III: Solo Beauty / Emotion"
    human_visible_prompt: str
    hidden_ar_world_state: Dict[str, str]
    dp_camera_and_lighting: str
    transition_to_next: str


@dataclass
class MusicVideoDiffusionPack:
    pack_id: str
    artist: str
    track_title: str
    year_and_director: str
    bpm_and_audio_spine: str
    core_diffusion_lesson: str
    three_movements_breakdown: Dict[str, str]
    shots: List[MusicVideoMovementShot] = field(default_factory=list)
    pure_diffusion_failure_mode: str = ""
    ar_hybrid_solution: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


GENERATIVE_PARADIGMS: List[GenerativeParadigmComparison] = [
    GenerativeParadigmComparison(
        paradigm_id="autoregressive",
        name="Autoregressive Models (AR)",
        cognitive_intuition="Looks a little bit behind, and then steps forward sequentially (P(x_t | x_<t)).",
        mathematical_form="P(X) = \\prod_{t=1}^{T} P(x_t \\mid x_1, x_2, \\dots, x_{t-1})",
        horse_in_grass_behavior="Commits token-by-token to an explicit symbolic ledger ('object=horse, coat=bay, background=paddock, fence=white_oak') and remembers it forever.",
        cinematic_strength="Causal continuity, object permanence, and off-screen actor tracking across 50+ cuts.",
        cinematic_blind_spot="Struggles to synthesize ultra-high-resolution continuous spatial textures and fluid lighting in parallel.",
    ),
    GenerativeParadigmComparison(
        paradigm_id="encoder_decoder",
        name="Encoder-Decoder Models (Seq2Seq)",
        cognitive_intuition="Looks both ways—wants to know everything in the input before emitting a high-confidence structured output.",
        mathematical_form="H = \\text{Encoder}(X_{1:N}), \\quad Y_{1:M} = \\text{Decoder}(H)",
        horse_in_grass_behavior="Reads the entire 3-minute song structure and screenplay first, then maps out global scene allocations with high confidence.",
        cinematic_strength="Global structural alignment (mapping a full song's verse/chorus/bridge waveform to a master edit decision list).",
        cinematic_blind_spot="Static structural translator; does not iteratively denoise visual manifolds.",
    ),
    GenerativeParadigmComparison(
        paradigm_id="pure_diffusion",
        name="Pure Diffusion Models (Latent Denoising)",
        cognitive_intuition="Propagates from Gaussian noise into underspecified creative space where human prompts leave room for visual imagination.",
        mathematical_form="p_\\theta(x_{t-1} \\mid x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t, c), \\Sigma_\\theta(x_t, t))",
        horse_in_grass_behavior="When asked for 'a horse in a grassy background', the human hasn't specified breed or field—so diffusion creatively samples a chestnut horse in a forest, lawn, or meadow. None of those are wrong!",
        cinematic_strength="Photorealistic spatial synthesis, atmospheric lighting, fabric/hair physics, and creative world-filling.",
        cinematic_blind_spot="Temporal Amnesia across cuts: in Shot 2, the church door changes color, the brass handle morphs into a knob, and the person behind the door vanishes.",
    ),
    GenerativeParadigmComparison(
        paradigm_id="ar_guided_diffusion",
        name="Autoregressive-Guided Diffusion (The Future of Cinema AI)",
        cognitive_intuition="An invisible Autoregressive Director maintains the persistent World-State Ledger in the background while Diffusion renders the pixels and DP transitions.",
        mathematical_form="z_{\\text{AR}}^{(t)} = \\text{AR}(z_{\\text{AR}}^{(<t)}, c_t, a_t), \\quad X_t \\sim \\text{Diffusion}(\\epsilon \\mid c_t, z_{\\text{AR}}^{(t)}, \\text{DP}_{t-1 \\to t})",
        horse_in_grass_behavior="Shot 1 freely imagines a dapple-grey horse in a misty meadow; the AR ledger immediately locks those exact traits so Shot 2's reverse angle preserves the exact horse, mist density, and fence post.",
        cinematic_strength="Combines photorealistic spatial imagination with 100% prop permanence, occluded-actor tracking, and beat-locked DP transitions.",
        cinematic_blind_spot="Requires cross-modal SFT training on tightly edited audiovisual corpora (such as music videos).",
    ),
]


CHURCH_DOOR_CONTINUITY_CASE = {
    "title": "The Church Door & Handle Invariant (Background Autoregressive World-State Outside the Human Prompt)",
    "premise": (
        "When a human prompts Shot 1 ('A door in front of a church') and Shot 2 ('The door opens'), "
        "the human should NOT have to manually re-specify 40 lines of hex colors and hardware specs in Shot 2. "
        "An Autoregressive World-State Model running in the background (outside the visible human prompt) "
        "must persist the physical invariants and occluded actors into the diffusion denoiser."
    ),
    "world_objects": [
        WorldStateObject(
            object_id="church_entrance_door_01",
            surface_attributes={
                "material": "weathered_crimson_stained_red_oak",
                "color_hex": "#5E1914",
                "arch_geometry": "gothic_pointed_lancet_arch",
                "handle_hardware": "heavy_oxidized_brass_ring_latch_left_mounted",
                "hinge_side": "right_hand_inward_swing",
            },
            occluded_or_offscreen_state={
                "actor_behind_door": "present (protagonist_in_crimson_velvet_cloak, right_hand_on_interior_iron_bar)",
                "interior_lighting": "3200K_warm_amber_candlelight_spill_with_incense_haze",
            },
            persistence_rule="LOCKED_ACROSS_ALL_CUTS: Any subsequent shot referencing 'the door' inherits color #5E1914, left brass ring latch, and reveals the occluded protagonist on swing > 15 degrees.",
        )
    ],
    "shot_sequence": [
        {
            "shot_number": 1,
            "human_prompt": "A wooden door in front of an ancient stone church at dusk.",
            "pure_diffusion_renders": "Crimson oak Gothic door (#5E1914) with a heavy oxidized brass ring handle on the left.",
            "hidden_ar_ledger_update": "COMMITTED -> door_color=#5E1914, handle=oxidized_brass_ring_left, occluded_actor=protagonist_waiting_inside",
            "ar_guided_diffusion_renders": "Crimson oak Gothic door (#5E1914) with a heavy oxidized brass ring handle on the left (State Hash: 0x9F4A).",
        },
        {
            "shot_number": 2,
            "human_prompt": "Close-up: the door opens.",
            "pure_diffusion_renders": "❌ DRIFT FAILURE: Renders a flat brown rectangular door (#8B5A2B) with a modern chrome lever on the right, opening into an empty dark void.",
            "hidden_ar_ledger_update": "INJECTED SILENTLY INTO CROSS-ATTENTION -> door_color=#5E1914 + handle=oxidized_brass_ring_left + reveal(occluded_actor=protagonist_in_crimson_velvet_cloak)",
            "ar_guided_diffusion_renders": "✅ CONTINUITY LOCKED: The exact #5E1914 crimson Gothic door unlatches via its left oxidized brass ring, spilling 3200K candlelight across the protagonist's hand as she steps forward.",
        },
    ],
}


MUSIC_VIDEO_PACKS: Dict[str, MusicVideoDiffusionPack] = {
    "britney_oops_i_did_it_again": MusicVideoDiffusionPack(
        pack_id="britney_oops_i_did_it_again",
        artist="Britney Spears",
        track_title="Oops!... I Did It Again",
        year_and_director="2000 · Dir. Nigel Dick · DP: Igor Jadue-Lillo",
        bpm_and_audio_spine="95 BPM · New Jack Swing / Dance-Pop · Tightly gated snare & spoken-word Titanic bridge",
        core_diffusion_lesson=(
            "Demonstrates the 3-Movement Pop/K-Pop Intercut Architecture: disparate visual worlds "
            "(Mars soundstage narrative, red latex catsuit group choreography, and spinning white-tunnel glamour solo) "
            "cut every 1.4 seconds without losing prop or wardrobe permanence, unified by DP whip-pans and the 95 BPM audio track."
        ),
        three_movements_breakdown={
            "Movement I (Narrative Story World)": "Red Mars surface soundstage with NASA astronaut, lunar rover, and the Heart of the Ocean diamond prop.",
            "Movement II (High-Energy Choreography Grid)": "Matte-black stage with geometric orange light panels, Britney in the iconic red latex catsuit with 8 synchronized backup dancers.",
            "Movement III (Glamour / Emotional Beauty Solo)": "Spinning futuristic white turbine tunnel & overhead butterfly-rig camera with Britney in white crop-top and metallic skirt.",
        },
        pure_diffusion_failure_mode=(
            "When cutting from Movement III (White Tunnel) back to Movement I (Mars), pure diffusion forgets the astronaut's silver visor reflection, "
            "alters the specular gloss of the red latex catsuit, and loses the 8-dancer spatial grid alignment."
        ),
        ar_hybrid_solution=(
            "The Autoregressive Ledger maintains 3 parallel scene registers (Reg_I_Mars, Reg_II_RedCatsuitGrid, Reg_III_WhiteTunnel) "
            "and conditions each 0.35s DP whip-pan on the 95 BPM snare transient."
        ),
        shots=[
            MusicVideoMovementShot(
                shot_id="oops_01",
                timecode="00:42.00 - 00:43.80",
                movement_layer="Movement I: Narrative World (Mars Surface)",
                human_visible_prompt="The astronaut uncovers a stone crate on Mars as Britney watches from the raised platform.",
                hidden_ar_world_state={
                    "wardrobe_reg_I": "red_patent_latex_catsuit_gloss_0.94",
                    "prop_state": "mars_crate_open + heart_of_the_ocean_jewel_occluded_in_astronaut_glove",
                    "mars_sky_gradient": "#C83E1B_terracotta_haze",
                },
                dp_camera_and_lighting="Low-angle 24mm anamorphic crane push-in; harsh overhead space-sun key light with crimson bounce.",
                transition_to_next="0.30s High-velocity upward whip-pan + white flash frame on beat 4 snare hit.",
            ),
            MusicVideoMovementShot(
                shot_id="oops_02",
                timecode="00:43.80 - 00:45.20",
                movement_layer="Movement II: Choreography Grid (Chorus Drop)",
                human_visible_prompt="Chorus hits: full-body synchronized arm-cross choreography with backup dancers.",
                hidden_ar_world_state={
                    "wardrobe_reg_II": "red_patent_latex_catsuit_gloss_0.94 (SHARED_INVARIANT_WITH_REG_I)",
                    "dancer_blocking": "8_dancers_V_formation_2.2m_spacing",
                    "audio_transient": "chorus_downbeat_95BPM_bar_17",
                },
                dp_camera_and_lighting="Centered 35mm dolly-zoom snap on downbeat; high-contrast rim lighting against matte-black floor.",
                transition_to_next="0.25s Match-action cut on Britney's right hand gesture matching her head turn in the white tunnel.",
            ),
            MusicVideoMovementShot(
                shot_id="oops_03",
                timecode="00:45.20 - 00:46.90",
                movement_layer="Movement III: Solo Beauty / Emotion (White Turbine Tunnel)",
                human_visible_prompt="Intimate close-up spinning overhead shot as she sings the hook directly into the lens.",
                hidden_ar_world_state={
                    "wardrobe_reg_III": "white_halter_crop_metallic_silver_pleated_skirt",
                    "camera_roll_angle": "continuous_counterclockwise_roll_45deg_per_sec",
                },
                dp_camera_and_lighting="50mm macro ring-light reflection in pupils; soft 5600K wrap-around beauty diffusion.",
                transition_to_next="0.35s Rotational match-cut back to Movement I (Mars) as the astronaut extends the Heart of the Ocean necklace.",
            ),
        ],
    ),
    "britney_baby_one_more_time": MusicVideoDiffusionPack(
        pack_id="britney_baby_one_more_time",
        artist="Britney Spears",
        track_title="...Baby One More Time",
        year_and_director="1998 · Dir. Nigel Dick · Venice High School, Los Angeles",
        bpm_and_audio_spine="93 BPM · Heavy Max Martin piano-bass slam & school-bell transient trigger",
        core_diffusion_lesson=(
            "Architectural & Environmental Progression Continuity: instead of cutting randomly, the 3 movements progress spatially "
            "from the claustrophobic classroom (Movement I) -> the locker hallway (Movement II) -> the outdoor courtyard & gymnasium (Movement III), "
            "while the AR state ledger tracks the school clock ticking to 3:00 PM and the reveal at the final bell."
        ),
        three_movements_breakdown={
            "Movement I (Classroom Boredom & Foot-Tap)": "Desks in rows, ticking wall clock approaching 3:00 PM, stern teacher at chalkboard, rhythmic pencil/foot tapping.",
            "Movement II (Hallway Locker Line Choreography)": "Green steel lockers, symmetrical 1-point perspective corridor, synchronized chair and locker-slam choreography.",
            "Movement III (Courtyard & Gymnasium Acrobatics)": "California afternoon sunlight, bleachers, basketball court backflips, and final return to the daydreaming classroom.",
        },
        pure_diffusion_failure_mode=(
            "Pure diffusion loses the 1-point perspective locker geometry, changes the grey cardigan knot height between shots, "
            "and forgets that the entire video is bookended by the classroom daydream."
        ),
        ar_hybrid_solution=(
            "The AR World-State Ledger tracks the global narrative frame (`is_daydream_sequence = True`, `clock_time = 14:59:58`) "
            "so the final cut snaps seamlessly back to the exact classroom desk lighting from Shot 1."
        ),
        shots=[
            MusicVideoMovementShot(
                shot_id="bomt_01",
                timecode="00:00.00 - 00:18.50",
                movement_layer="Movement I: Narrative World (Classroom Clock Countdown)",
                human_visible_prompt="Close-up of a classroom clock ticking to 3:00 PM as Britney taps her sneaker on the floor.",
                hidden_ar_world_state={"prop_clock": "analog_school_clock_02:59:59", "wardrobe_reg": "grey_cardigan_white_shirt_pink_pom_poms"},
                dp_camera_and_lighting="Dutch-angle 28mm push-in on desk; fluorescent 4100K classroom overhead mixed with warm window key.",
                transition_to_next="Audio transient cut: dielectric school-bell ring + 3-note bass piano slam launches hallway doors open.",
            ),
            MusicVideoMovementShot(
                shot_id="bomt_02",
                timecode="00:18.50 - 00:45.00",
                movement_layer="Movement II: Choreography Grid (Venice High Locker Corridor)",
                human_visible_prompt="Synchronized dance formation down the school hallway flanked by green lockers.",
                hidden_ar_world_state={"corridor_geometry": "1_point_perspective_olive_green_lockers_#3B4D3A", "wardrobe_reg": "grey_cardigan_white_shirt_pink_pom_poms"},
                dp_camera_and_lighting="Low Steadicam backwards tracking shot at 24fps; specular floor reflections.",
                transition_to_next="Whip-tilt on locker slam into bright exterior California sunlight (Movement III).",
            ),
        ],
    ),
    "decemberists_o_valencia": MusicVideoDiffusionPack(
        pack_id="decemberists_o_valencia",
        artist="The Decemberists",
        track_title="O Valencia!",
        year_and_director="2006 · Dir. Aaron Stewart-Ahn · Folk-Rock Romeo & Juliet Gang Tragedy",
        bpm_and_audio_spine="134 BPM · Driving acoustic guitar strum & tragic narrative ballad pacing",
        core_diffusion_lesson=(
            "Long-Arc Narrative Causality & Prop Consequence: unlike pure pop performance videos, 'O Valencia!' tells a linear tragic story "
            "of star-crossed lovers from rival gangs. An Autoregressive-Diffusion model must track character allegiance tokens, the hotel window standoff, "
            "and the irreversible causal consequence of the fatal gunshot in the final act."
        ),
        three_movements_breakdown={
            "Movement I (Rival Territory Standoff)": "Two rival gangs distinguished by strict color/wardrobe tokens confronting each other across urban turf.",
            "Movement II (The Forbidden Rendezvous & Escape)": "Secret meeting, motorcycle/getaway urgency, and the band's deadpan narrative performance intercuts.",
            "Movement III (The Tragic Climax)": "The confrontation, the crossfire leap into the bullet's path, and the mournful aftermath.",
        },
        pure_diffusion_failure_mode=(
            "Pure diffusion swaps gang wardrobe insignias between cuts, breaking the viewer's ability to tell which faction is which."
        ),
        ar_hybrid_solution=(
            "The AR Ledger enforces strict faction-entity bindings (`Faction_A_Brother`, `Faction_B_Protagonist`, `Valencia_Heroine`) across every cross-cut."
        ),
        shots=[],
    ),
    "michael_jackson_thriller": MusicVideoDiffusionPack(
        pack_id="michael_jackson_thriller",
        artist="Michael Jackson",
        track_title="Thriller",
        year_and_director="1983 · Dir. John Landis · DP: Robert Paynter (35mm Film)",
        bpm_and_audio_spine="118 BPM · Quincy Jones / Rod Temperton synth-funk bassline & Vincent Price spoken bridge",
        core_diffusion_lesson=(
            "Nested Diegetic Reality Layers & Morphological State Persistence: 'Thriller' switches across 3 distinct reality layers "
            "(1950s movie-within-a-movie werewolf transformation -> 1980s cinema-street courtship & zombie choreography -> haunted house climax). "
            "The AR ledger must track which reality layer is active while preserving the red leather V-jacket geometry across human and zombie states!"
        ),
        three_movements_breakdown={
            "Movement I (1950s Film-Within-A-Film Metamorphosis)": "Moonlight reveal, practical Rick Baker werewolf facial morphing, and theater audience reaction cut.",
            "Movement II (Foggy Street Walk & Undead Choreography)": "Iconic red-and-black V-chevron leather jacket, streetlamp fog, and the synchronized zombie troupe dance.",
            "Movement III (Haunted House Siege & Freeze-Frame Yellow Eyes)": "Interior floorboards bursting, sudden match-cut back to waking human reality, and the final 4th-wall yellow-iris turn.",
        },
        pure_diffusion_failure_mode=(
            "When transitioning from human Michael to zombie Michael and back inside the haunted house, pure diffusion mangles the red V-jacket chevron angles "
            "and fails the instant match-cut from surrounding zombies back to the calm living-room hand-reach."
        ),
        ar_hybrid_solution=(
            "The AR World-State Ledger holds a shared pose skeleton (`pose_arm_extended_32deg`) while flipping the environment & makeup state token "
            "from `State_Undead_Siege` to `State_Living_Room_Calm` on the exact frame of the match-cut."
        ),
        shots=[],
    ),
    "michael_jackson_remember_the_time": MusicVideoDiffusionPack(
        pack_id="michael_jackson_remember_the_time",
        artist="Michael Jackson",
        track_title="Remember the Time",
        year_and_director="1992 · Dir. John Singleton · Feat. Eddie Murphy, Iman & Magic Johnson",
        bpm_and_audio_spine="108 BPM · Teddy Riley New Jack Swing percussion & Egyptian finger-cymbal groove",
        core_diffusion_lesson=(
            "Literal Particle-Diffusion Metaphor (Golden Sand Swirl -> Coherent Human Form): In 'Remember the Time', the hooded stranger "
            "dissolves into a swirling vortex of golden sand particles and reforms into Michael Jackson in gold Pharaonic armor—a literal 1992 visual precursor "
            "to modern forward-noising and reverse-denoising diffusion trajectories!"
        ),
        three_movements_breakdown={
            "Movement I (Pharaoh's Court & Particle Materialization)": "Throne room of Ramses (Eddie Murphy) and Queen Nefertiti (Iman); the hooded figure noisifies into golden sand and denoises into Michael.",
            "Movement II (Palace Guard Pursuit & Market Chase)": "Whip-pan corridors, market stalls, and athletic evasion cuts.",
            "Movement III (The Golden Egyptian Precision Breakdown)": "Complex polyrhythmic group choreography in gold-and-turquoise pleated kilts before vanishing back into a sandstorm.",
        },
        pure_diffusion_failure_mode=(
            "Standard video models treat sand dispersion as random fluid noise and lose conservation of golden mass and lighting reflections on the throne room pillars."
        ),
        ar_hybrid_solution=(
            "Couples the reverse-diffusion denoising schedule t: T -> 0 directly to the choreographed sand-vortex keyframes while locking the Pharaoh's throne-room camera matrix."
        ),
        shots=[],
    ),
    "miley_cyrus_wrecking_ball": MusicVideoDiffusionPack(
        pack_id="miley_cyrus_wrecking_ball",
        artist="Miley Cyrus",
        track_title="Wrecking Ball",
        year_and_director="2013 · Dir. Terry Richardson",
        bpm_and_audio_spine="120 BPM · Stark verse piano minimal build -> Massive distorted chorus sub-bass drop",
        core_diffusion_lesson=(
            "Extreme Dynamic Range & Irreversible Physics Destruction Ledger: Contrast between static, hyper-intimate emotional micro-expression "
            "(Movement I: tight white-background facial close-up with real tears) and violent structural physics (Movement II: swinging iron wrecking ball "
            "shattering brick walls). Once the wall is smashed in Chorus 1, the AR ledger must keep the rubble on the floor in Verse 2!"
        ),
        three_movements_breakdown={
            "Movement I (Sinead O'Connor Homage — Intimate Macro Portrait)": "Unblinking direct-to-lens close-up against a stark white background; micro-tremors of lip and tear ducts during quiet verse.",
            "Movement II (Pendulum Momentum & Structural Demolition)": "Wide high-speed shots of the steel wrecking ball pendulum smashing through cinderblock walls in slow motion.",
            "Movement III (Aftermath Rubble & Emotional Exhaustion)": "Grounded shots amidst the shattered masonry dust as the chorus subsides.",
        },
        pure_diffusion_failure_mode=(
            "In pure diffusion, the demolished brick wall magically repairs itself between cuts, and the tear tracks on the singer's cheek jump positions."
        ),
        ar_hybrid_solution=(
            "The AR Physics State Ledger records irreversible entropy (`wall_north_integrity = 0.0_demolished_rubble_pile_seed_42`) so every subsequent cut respects the shattered set."
        ),
        shots=[],
    ),
    "elton_john_your_song": MusicVideoDiffusionPack(
        pack_id="elton_john_your_song",
        artist="Elton John",
        track_title="Your Song",
        year_and_director="1970 / Classic Piano Performance & Lyrical Continuous-Drift Archetype",
        bpm_and_audio_spine="64 BPM · Rubato acoustic piano arpeggios & intimate conversational vocal delivery",
        core_diffusion_lesson=(
            "The Long-Take Lyrical Counterpoint to Rapid Montage: When the song is a gentle 64 BPM acoustic ballad, the DP grammar shifts from "
            "0.4s whip-pans to slow, breathing 18-second continuous dolly arcs around the grand piano, where subtle eye-line shifts and piano-key finger placement must remain drift-free."
        ),
        three_movements_breakdown={
            "Movement I (Intimate Piano Hands & Sheet Music Establishing)": "Warm tungsten spotlight on ivory keys, subtle pedal mechanics, and reflective lacquer.",
            "Movement II (Slow 180-Degree Orbital Dolly Around Vocalist)": "Continuous camera parallax revealing studio depth and ambient dust motes.",
            "Movement III (Emotional Chorus Swell & Direct Eye-Line Resolve)": "Gradual focal-length compression into an intimate portrait as the strings enter.",
        },
        pure_diffusion_failure_mode=(
            "Over a slow 18-second orbital dolly, pure diffusion suffers from 'structural boiling'—the grand piano grows extra octaves of keys and the microphone stand drifts."
        ),
        ar_hybrid_solution=(
            "The AR Spatial Anchor locks the 3D bounding geometry of the 88-key grand piano and microphone boom while diffusion renders the warm 35mm film grain and lens halation."
        ),
        shots=[],
    ),
    "elton_john_candle_in_the_wind_1973_vs_1997": MusicVideoDiffusionPack(
        pack_id="elton_john_candle_in_the_wind_1973_vs_1997",
        artist="Elton John (Bernie Taupin)",
        track_title="Candle in the Wind (1973 Marilyn Monroe Original vs. 1997 Princess Diana Westminster Abbey Re-Release)",
        year_and_director="1973 (Goodbye Yellow Brick Road) vs. 1997 (Goodbye England's Rose — Westminster Abbey Live)",
        bpm_and_audio_spine="63 BPM · Identical E-Major Acoustic Piano Chord Progression & Melodic Spine Across Both Eras",
        core_diffusion_lesson=(
            "SFT Context-Transfer Across Identical Audio Waveforms: The 1973 and 1997 versions share the exact same underlying musical chord "
            "progression and tempo, yet swapping the conditioning era/subject token ('Norma Jeane / 1950s Hollywood Flashbulbs' vs. 'England's Rose / "
            "1997 Westminster Abbey Candlelit Mourning') completely re-maps the visual diffusion manifold, color temperature, and camera solemnity!"
        ),
        three_movements_breakdown={
            "Movement I (1973 Conditioned — 1950s Hollywood Press & Silver Screen Tragedy)": "High-contrast black-and-white / Kodachrome flashbulbs, red carpet press cameras, and lonely dressing-room mirror reflections.",
            "Movement II (1997 Conditioned — Westminster Abbey Cathedral Solemnity)": "Gothic stone arches, thousands of flickering beeswax memorials outside Kensington Palace, and restrained stationary broadcast camera framing.",
            "Movement III (Shared Acoustic Invariant — The Solitary Grand Piano)": "The invariant anchor across both 1973 and 1997: Elton John at the grand piano, proving how one musical latent spine can drive two radically different historical visual worlds.",
        },
        pure_diffusion_failure_mode=(
            "An unconditioned audio-to-video diffusion model listening only to the E-major piano track blends incompatible 1970s rock-stage aesthetics with solemn state-funeral imagery."
        ),
        ar_hybrid_solution=(
            "Supervised Fine-Tuning (SFT) binds an explicit Historical Era & Subject Conditioning Token (`<|ERA:1973_HOLLYWOOD_ELEGY|>` vs. `<|ERA:1997_WESTMINSTER_ROSE|>`) "
            "to the shared audio latent spine, shifting lighting, lens velocity, and crowd posture deterministically."
        ),
        shots=[],
    ),
}


def list_mv_packs() -> List[Dict[str, str]]:
    """Return summary metadata for all Music Video Autoregressive-Diffusion Case Studies."""
    return [
        {
            "pack_id": p.pack_id,
            "artist": p.artist,
            "track_title": p.track_title,
            "bpm_and_audio_spine": p.bpm_and_audio_spine,
            "core_diffusion_lesson": p.core_diffusion_lesson,
        }
        for p in MUSIC_VIDEO_PACKS.values()
    ]


def get_mv_pack(pack_id: str) -> MusicVideoDiffusionPack:
    """Retrieve a specific MusicVideoDiffusionPack by ID."""
    if pack_id not in MUSIC_VIDEO_PACKS:
        available = ", ".join(MUSIC_VIDEO_PACKS.keys())
        raise KeyError(f"Unknown pack_id '{pack_id}'. Available packs: {available}")
    return MUSIC_VIDEO_PACKS[pack_id]
