def forward_256(self,
    TemplateDistogram: Tensor,
    TemplateMask: Tensor,
    TemplateResType: Tensor,
    TemplateUnitVector: Tensor,
    ChainIsCropped: Tensor,
    ESMEmbeddings: Tensor,
    IsDistillation: Tensor,
    MSADeletionMean: Tensor,
    MSAProfile: Tensor,
    MissingChainContact: Tensor,
    ResidueType: Tensor,
    TokenBFactor: Tensor,
    TokenPLDDT: Tensor,
    BlockedAtomPairDistogram: Tensor,
    InverseSquaredBlockedAtomPairDistances: Tensor,
    IsPairedMSA: Tensor,
    MSADataSource: Tensor,
    MSADeletionValue: Tensor,
    MSAHasDeletion: Tensor,
    MSAOneHot: Tensor,
    DockingConstraintGenerator: Tensor,
    RelativeChain: Tensor,
    RelativeEntity: Tensor,
    RelativeSequenceSeparation: Tensor,
    RelativeTokenSeparation: Tensor,
    TokenDistanceRestraint: Tensor,
    TokenPairPocketRestraint: Tensor,
    AtomNameOneHot: Tensor,
    AtomRefCharge: Tensor,
    AtomRefElement: Tensor,
    AtomRefMask: Tensor,
    AtomRefPos: Tensor) -> Dict[str, Tensor]:
  input_projs = self.input_projs
  TEMPLATES = input_projs.TEMPLATES
  _0 = getattr(TEMPLATES, "0")
  feature_embeddings = self.feature_embeddings
  TEMPLATES0 = feature_embeddings.TEMPLATES
  TemplateResType0 = TEMPLATES0.TemplateResType
  embedding = TemplateResType0.embedding
  input_projs0 = self.input_projs
  MSA = input_projs0.MSA
  _00 = getattr(MSA, "0")
  input_projs1 = self.input_projs
  TOKEN_PAIR = input_projs1.TOKEN_PAIR
  _01 = getattr(TOKEN_PAIR, "0")
  feature_embeddings0 = self.feature_embeddings
  TOKEN_PAIR0 = feature_embeddings0.TOKEN_PAIR
  TokenPairPocketRestraint0 = TOKEN_PAIR0.TokenPairPocketRestraint
  radii = TokenPairPocketRestraint0.radii
  feature_embeddings1 = self.feature_embeddings
  TOKEN_PAIR1 = feature_embeddings1.TOKEN_PAIR
  TokenDistanceRestraint0 = TOKEN_PAIR1.TokenDistanceRestraint
  radii0 = TokenDistanceRestraint0.radii
  input_projs2 = self.input_projs
  TOKEN = input_projs2.TOKEN
  _02 = getattr(TOKEN, "0")
  input_projs3 = self.input_projs
  ATOM_PAIR = input_projs3.ATOM_PAIR
  _03 = getattr(ATOM_PAIR, "0")
  input_projs4 = self.input_projs
  ATOM = input_projs4.ATOM
  _04 = getattr(ATOM, "0")
  _5 = torch.one_hot(torch.to(AtomNameOneHot, 4), 65)
  embedded_feat = torch.to(_5, 6)
  _1 = ops.prim.NumToTensor(torch.size(AtomNameOneHot, 0))
  _2 = int(_1)
  _3 = ops.prim.NumToTensor(torch.size(AtomNameOneHot, 1))
  _4 = torch.reshape(embedded_feat, [_2, int(_3), 260])
  _6 = ops.prim.NumToTensor(torch.size(AtomRefCharge, 0))
  _7 = int(_6)
  _8 = ops.prim.NumToTensor(torch.size(AtomRefCharge, 1))
  _9 = torch.reshape(AtomRefCharge, [_7, int(_8), 1])
  _10 = torch.one_hot(torch.to(AtomRefElement, 4), 130)
  embedded_feat0 = torch.to(_10, 6)
  _11 = ops.prim.NumToTensor(torch.size(AtomRefElement, 0))
  _12 = int(_11)
  _13 = ops.prim.NumToTensor(torch.size(AtomRefElement, 1))
  _14 = torch.reshape(embedded_feat0, [_12, int(_13), 130])
  _15 = ops.prim.NumToTensor(torch.size(AtomRefMask, 0))
  _16 = int(_15)
  _17 = ops.prim.NumToTensor(torch.size(AtomRefMask, 1))
  _18 = torch.reshape(AtomRefMask, [_16, int(_17), 1])
  _19 = ops.prim.NumToTensor(torch.size(AtomRefPos, 0))
  _20 = int(_19)
  _21 = ops.prim.NumToTensor(torch.size(AtomRefPos, 1))
  _22 = torch.reshape(AtomRefPos, [_20, int(_21), 3])
  input = torch.cat([_4, _9, _14, _18, _22], -1)
  _23 = (_04).forward(input, )
  _24 = torch.one_hot(torch.to(BlockedAtomPairDistogram, 4), 12)
  embedded_feat1 = torch.to(_24, 6)
  _25 = torch.size(BlockedAtomPairDistogram, 0)
  _26 = int(ops.prim.NumToTensor(_25))
  _27 = torch.size(BlockedAtomPairDistogram, 1)
  _28 = int(ops.prim.NumToTensor(_27))
  _29 = torch.size(BlockedAtomPairDistogram, 2)
  _30 = int(ops.prim.NumToTensor(_29))
  _31 = torch.size(BlockedAtomPairDistogram, 3)
  _32 = [_26, _28, _30, int(ops.prim.NumToTensor(_31)), 12]
  _33 = torch.reshape(embedded_feat1, _32)
  _34 = torch.size(InverseSquaredBlockedAtomPairDistances, 0)
  _35 = int(ops.prim.NumToTensor(_34))
  _36 = torch.size(InverseSquaredBlockedAtomPairDistances, 1)
  _37 = int(ops.prim.NumToTensor(_36))
  _38 = torch.size(InverseSquaredBlockedAtomPairDistances, 2)
  _39 = int(ops.prim.NumToTensor(_38))
  _40 = torch.size(InverseSquaredBlockedAtomPairDistances, 3)
  _41 = [_35, _37, _39, int(ops.prim.NumToTensor(_40)), 2]
  _42 = torch.reshape(InverseSquaredBlockedAtomPairDistances, _41)
  input0 = torch.cat([_33, _42], -1)
  _43 = (_03).forward(input0, )
  _44 = ops.prim.NumToTensor(torch.size(ChainIsCropped, 0))
  _45 = int(_44)
  _46 = ops.prim.NumToTensor(torch.size(ChainIsCropped, 1))
  _47 = torch.reshape(ChainIsCropped, [_45, int(_46), 1])
  _48 = ops.prim.NumToTensor(torch.size(ESMEmbeddings, 0))
  _49 = int(_48)
  _50 = ops.prim.NumToTensor(torch.size(ESMEmbeddings, 1))
  _51 = torch.reshape(ESMEmbeddings, [_49, int(_50), 2560])
  _52 = torch.one_hot(torch.to(IsDistillation, 4), 2)
  embedded_feat2 = torch.to(_52, 6)
  _53 = ops.prim.NumToTensor(torch.size(IsDistillation, 0))
  _54 = int(_53)
  _55 = ops.prim.NumToTensor(torch.size(IsDistillation, 1))
  _56 = torch.reshape(embedded_feat2, [_54, int(_55), 2])
  _57 = ops.prim.NumToTensor(torch.size(MSADeletionMean, 0))
  _58 = int(_57)
  _59 = ops.prim.NumToTensor(torch.size(MSADeletionMean, 1))
  _60 = torch.reshape(MSADeletionMean, [_58, int(_59), 1])
  _61 = ops.prim.NumToTensor(torch.size(MSAProfile, 0))
  _62 = int(_61)
  _63 = ops.prim.NumToTensor(torch.size(MSAProfile, 1))
  _64 = torch.reshape(MSAProfile, [_62, int(_63), 33])
  _65 = ops.prim.NumToTensor(torch.size(MissingChainContact, 0))
  _66 = int(_65)
  _67 = ops.prim.NumToTensor(torch.size(MissingChainContact, 1))
  _68 = torch.reshape(MissingChainContact, [_66, int(_67), 1])
  _69 = torch.one_hot(torch.to(ResidueType, 4), 33)
  embedded_feat3 = torch.to(_69, 6)
  _70 = ops.prim.NumToTensor(torch.size(ResidueType, 0))
  _71 = int(_70)
  _72 = ops.prim.NumToTensor(torch.size(ResidueType, 1))
  _73 = torch.reshape(embedded_feat3, [_71, int(_72), 33])
  _74 = torch.one_hot(torch.to(TokenBFactor, 4), 3)
  embedded_feat4 = torch.to(_74, 6)
  _75 = ops.prim.NumToTensor(torch.size(TokenBFactor, 0))
  _76 = int(_75)
  _77 = ops.prim.NumToTensor(torch.size(TokenBFactor, 1))
  _78 = torch.reshape(embedded_feat4, [_76, int(_77), 3])
  _79 = torch.one_hot(torch.to(TokenPLDDT, 4), 4)
  embedded_feat5 = torch.to(_79, 6)
  _80 = ops.prim.NumToTensor(torch.size(TokenPLDDT, 0))
  _81 = int(_80)
  _82 = ops.prim.NumToTensor(torch.size(TokenPLDDT, 1))
  _83 = torch.reshape(embedded_feat5, [_81, int(_82), 4])
  _84 = [_47, _51, _56, _60, _64, _68, _73, _78, _83]
  input1 = torch.cat(_84, -1)
  _85 = (_02).forward(input1, )
  _86 = torch.to(DockingConstraintGenerator, 4)
  embedded_feat6 = torch.to(torch.one_hot(_86, 6), 6)
  _87 = torch.size(DockingConstraintGenerator, 0)
  _88 = int(ops.prim.NumToTensor(_87))
  _89 = torch.size(DockingConstraintGenerator, 1)
  _90 = int(ops.prim.NumToTensor(_89))
  _91 = torch.size(DockingConstraintGenerator, 2)
  _92 = [_88, _90, int(ops.prim.NumToTensor(_91)), 6]
  _93 = torch.reshape(embedded_feat6, _92)
  _94 = torch.one_hot(torch.to(RelativeChain, 4), 6)
  embedded_feat7 = torch.to(_94, 6)
  _95 = ops.prim.NumToTensor(torch.size(RelativeChain, 0))
  _96 = int(_95)
  _97 = ops.prim.NumToTensor(torch.size(RelativeChain, 1))
  _98 = int(_97)
  _99 = ops.prim.NumToTensor(torch.size(RelativeChain, 2))
  _100 = torch.reshape(embedded_feat7, [_96, _98, int(_99), 6])
  _101 = torch.one_hot(torch.to(RelativeEntity, 4), 3)
  embedded_feat8 = torch.to(_101, 6)
  _102 = ops.prim.NumToTensor(torch.size(RelativeEntity, 0))
  _103 = int(_102)
  _104 = ops.prim.NumToTensor(torch.size(RelativeEntity, 1))
  _105 = int(_104)
  _106 = ops.prim.NumToTensor(torch.size(RelativeEntity, 2))
  _107 = torch.reshape(embedded_feat8, [_103, _105, int(_106), 3])
  _108 = torch.to(RelativeSequenceSeparation, 4)
  embedded_feat9 = torch.to(torch.one_hot(_108, 67), 6)
  _109 = torch.size(RelativeSequenceSeparation, 0)
  _110 = int(ops.prim.NumToTensor(_109))
  _111 = torch.size(RelativeSequenceSeparation, 1)
  _112 = int(ops.prim.NumToTensor(_111))
  _113 = torch.size(RelativeSequenceSeparation, 2)
  _114 = [_110, _112, int(ops.prim.NumToTensor(_113)), 67]
  _115 = torch.reshape(embedded_feat9, _114)
  _116 = torch.one_hot(torch.to(RelativeTokenSeparation, 4), 67)
  embedded_feat10 = torch.to(_116, 6)
  _117 = torch.size(RelativeTokenSeparation, 0)
  _118 = int(ops.prim.NumToTensor(_117))
  _119 = torch.size(RelativeTokenSeparation, 1)
  _120 = int(ops.prim.NumToTensor(_119))
  _121 = torch.size(RelativeTokenSeparation, 2)
  _122 = [_118, _120, int(ops.prim.NumToTensor(_121)), 67]
  _123 = torch.reshape(embedded_feat10, _122)
  raw_data = torch.unsqueeze(TokenDistanceRestraint, -1)
  _124 = torch.unsqueeze(torch.unsqueeze(radii0, 0), 1)
  radii1 = torch.unsqueeze(_124, 2)
  _125 = torch.div(torch.sub(radii1, raw_data), CONSTANTS.c0)
  exp_val = torch.square(_125)
  clamped_exp_val = torch.clamp_max(exp_val, 16)
  encoding = torch.exp(torch.neg(clamped_exp_val))
  _126 = annotate(List[Optional[Tensor]], [torch.eq(clamped_exp_val, 16)])
  encoding0 = torch.index_put_(encoding, _126, CONSTANTS.c1)
  should_mask = torch.to(torch.eq(raw_data, -1.), 6)
  encoding1 = torch.mul(encoding0, torch.rsub(should_mask, 1))
  embedded_feat11 = torch.cat([encoding1, should_mask], -1)
  _127 = ops.prim.NumToTensor(torch.size(TokenDistanceRestraint, 0))
  _128 = int(_127)
  _129 = ops.prim.NumToTensor(torch.size(TokenDistanceRestraint, 1))
  _130 = int(_129)
  _131 = ops.prim.NumToTensor(torch.size(TokenDistanceRestraint, 2))
  _132 = torch.reshape(embedded_feat11, [_128, _130, int(_131), 7])
  raw_data0 = torch.unsqueeze(TokenPairPocketRestraint, -1)
  _133 = torch.unsqueeze(torch.unsqueeze(radii, 0), 1)
  radii2 = torch.unsqueeze(_133, 2)
  _134 = torch.div(torch.sub(radii2, raw_data0), CONSTANTS.c2)
  exp_val0 = torch.square(_134)
  clamped_exp_val0 = torch.clamp_max(exp_val0, 16)
  encoding2 = torch.exp(torch.neg(clamped_exp_val0))
  _135 = annotate(List[Optional[Tensor]], [torch.eq(clamped_exp_val0, 16)])
  encoding3 = torch.index_put_(encoding2, _135, CONSTANTS.c1)
  should_mask0 = torch.to(torch.eq(raw_data0, -1.), 6)
  encoding4 = torch.mul(encoding3, torch.rsub(should_mask0, 1))
  embedded_feat12 = torch.cat([encoding4, should_mask0], -1)
  _136 = torch.size(TokenPairPocketRestraint, 0)
  _137 = int(ops.prim.NumToTensor(_136))
  _138 = torch.size(TokenPairPocketRestraint, 1)
  _139 = int(ops.prim.NumToTensor(_138))
  _140 = torch.size(TokenPairPocketRestraint, 2)
  _141 = [_137, _139, int(ops.prim.NumToTensor(_140)), 7]
  _142 = [_93, _100, _107, _115, _123, _132, torch.reshape(embedded_feat12, _141)]
  input2 = torch.cat(_142, -1)
  _143 = (_01).forward(input2, )
  _144 = ops.prim.NumToTensor(torch.size(IsPairedMSA, 0))
  _145 = int(_144)
  _146 = ops.prim.NumToTensor(torch.size(IsPairedMSA, 1))
  _147 = int(_146)
  _148 = ops.prim.NumToTensor(torch.size(IsPairedMSA, 2))
  _149 = torch.reshape(IsPairedMSA, [_145, _147, int(_148), 1])
  _150 = torch.one_hot(torch.to(MSADataSource, 4), 6)
  embedded_feat13 = torch.to(_150, 6)
  _151 = ops.prim.NumToTensor(torch.size(MSADataSource, 0))
  _152 = int(_151)
  _153 = ops.prim.NumToTensor(torch.size(MSADataSource, 1))
  _154 = int(_153)
  _155 = ops.prim.NumToTensor(torch.size(MSADataSource, 2))
  _156 = torch.reshape(embedded_feat13, [_152, _154, int(_155), 6])
  _157 = ops.prim.NumToTensor(torch.size(MSADeletionValue, 0))
  _158 = int(_157)
  _159 = ops.prim.NumToTensor(torch.size(MSADeletionValue, 1))
  _160 = int(_159)
  _161 = ops.prim.NumToTensor(torch.size(MSADeletionValue, 2))
  _162 = torch.reshape(MSADeletionValue, [_158, _160, int(_161), 1])
  _163 = ops.prim.NumToTensor(torch.size(MSAHasDeletion, 0))
  _164 = int(_163)
  _165 = ops.prim.NumToTensor(torch.size(MSAHasDeletion, 1))
  _166 = int(_165)
  _167 = ops.prim.NumToTensor(torch.size(MSAHasDeletion, 2))
  _168 = torch.reshape(MSAHasDeletion, [_164, _166, int(_167), 1])
  _169 = torch.one_hot(torch.to(MSAOneHot, 4), 33)
  embedded_feat14 = torch.to(_169, 6)
  _170 = ops.prim.NumToTensor(torch.size(MSAOneHot, 0))
  _171 = int(_170)
  _172 = ops.prim.NumToTensor(torch.size(MSAOneHot, 1))
  _173 = int(_172)
  _174 = ops.prim.NumToTensor(torch.size(MSAOneHot, 2))
  _175 = torch.reshape(embedded_feat14, [_171, _173, int(_174), 33])
  input3 = torch.cat([_149, _156, _162, _168, _175], -1)
  _176 = (_00).forward(input3, )
  _177 = torch.one_hot(torch.to(TemplateDistogram, 4), 39)
  embedded_feat15 = torch.to(_177, 6)
  _178 = ops.prim.NumToTensor(torch.size(TemplateDistogram, 0))
  _179 = int(_178)
  _180 = ops.prim.NumToTensor(torch.size(TemplateDistogram, 1))
  _181 = int(_180)
  _182 = ops.prim.NumToTensor(torch.size(TemplateDistogram, 2))
  _183 = int(_182)
  _184 = ops.prim.NumToTensor(torch.size(TemplateDistogram, 3))
  _185 = torch.reshape(embedded_feat15, [_179, _181, _183, int(_184), 39])
  _186 = ops.prim.NumToTensor(torch.size(TemplateMask, 0))
  _187 = int(_186)
  _188 = ops.prim.NumToTensor(torch.size(TemplateMask, 1))
  _189 = int(_188)
  _190 = ops.prim.NumToTensor(torch.size(TemplateMask, 2))
  _191 = int(_190)
  _192 = ops.prim.NumToTensor(torch.size(TemplateMask, 3))
  _193 = torch.reshape(TemplateMask, [_187, _189, _191, int(_192), 2])
  to_emb = torch.to(TemplateResType, 4)
  size = ops.prim.NumToTensor(torch.size(to_emb, 3))
  _194 = annotate(number, size)
  out = torch.new_empty(to_emb, [int(size)], dtype=4, layout=None, device=None, pin_memory=False)
  offsets = torch.mul(torch.arange(0, _194, out=out), CONSTANTS.c3)
  input4 = torch.add(to_emb, offsets)
  _195 = (embedding).forward(input4, )
  element = ops.prim.NumToTensor(torch.size(_195, 0))
  element0 = ops.prim.NumToTensor(torch.size(_195, 1))
  element1 = ops.prim.NumToTensor(torch.size(_195, 2))
  element2 = ops.prim.NumToTensor(torch.size(_195, 3))
  element3 = ops.prim.NumToTensor(torch.size(_195, 4))
  _196 = int(torch.mul(element, CONSTANTS.c4))
  _197 = int(torch.mul(element0, CONSTANTS.c4))
  _198 = int(torch.mul(element1, CONSTANTS.c4))
  result = torch.mul(element2, CONSTANTS.c4)
  _199 = [_196, _197, _198, 1, int(torch.mul_(result, element3))]
  _200 = torch.reshape(_195, _199)
  element4 = ops.prim.NumToTensor(torch.size(_195, 0))
  element5 = ops.prim.NumToTensor(torch.size(_195, 1))
  element6 = ops.prim.NumToTensor(torch.size(_195, 2))
  element7 = ops.prim.NumToTensor(torch.size(_195, 3))
  element8 = ops.prim.NumToTensor(torch.size(_195, 4))
  _201 = int(torch.mul(element4, CONSTANTS.c4))
  _202 = int(torch.mul(element5, CONSTANTS.c4))
  _203 = int(torch.mul(element6, CONSTANTS.c4))
  result0 = torch.mul(element7, CONSTANTS.c4)
  _204 = [_201, _202, 1, _203, int(torch.mul_(result0, element8))]
  emb = torch.add(_200, torch.reshape(_195, _204))
  _205 = ops.prim.NumToTensor(torch.size(TemplateResType, 0))
  _206 = int(_205)
  _207 = ops.prim.NumToTensor(torch.size(TemplateResType, 1))
  _208 = int(_207)
  _209 = ops.prim.NumToTensor(torch.size(TemplateResType, 2))
  _210 = int(_209)
  _211 = ops.prim.NumToTensor(torch.size(TemplateResType, 2))
  _212 = torch.reshape(emb, [_206, _208, _210, int(_211), 32])
  _213 = ops.prim.NumToTensor(torch.size(TemplateUnitVector, 0))
  _214 = int(_213)
  _215 = ops.prim.NumToTensor(torch.size(TemplateUnitVector, 1))
  _216 = int(_215)
  _217 = ops.prim.NumToTensor(torch.size(TemplateUnitVector, 2))
  _218 = int(_217)
  _219 = ops.prim.NumToTensor(torch.size(TemplateUnitVector, 3))
  _220 = torch.reshape(TemplateUnitVector, [_214, _216, _218, int(_219), 3])
  input5 = torch.cat([_185, _193, _212, _220], -1)
  _221 = {"ATOM": _23, "ATOM_PAIR": _43, "TOKEN": _85, "TOKEN_PAIR": _143, "MSA": _176, "TEMPLATES": (_0).forward(input5, )}
  return _221
