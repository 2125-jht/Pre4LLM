#include "matxscript/runtime/codegen_all_includes.h"
#include <math.h>

using namespace ::matxscript::runtime;
extern "C" void* __matxscript_module_ctx = NULL;

extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__;

extern "C" MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__FunctionSet;
namespace {
// User class forward declarations
struct FunctionSet;
struct FunctionSet_SharedView;

FunctionSet_SharedView FunctionSet__F___init___wrapper(void* handle_2_71828182846=((void*)(int64_t)0));
int FunctionSet__F___init___wrapper__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL RTValue FunctionSet__F___init__(const FunctionSet_SharedView& self, void* handle_2_71828182846=((void*)(int64_t)0));
int FunctionSet__F___init____c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
MATX_DLL RTValue FunctionSet__F_get_score(const FunctionSet_SharedView& self, const DragonflyContext_SharedView& ctx);
int FunctionSet__F_get_score__c_api(MATXScriptAny*, int, MATXScriptAny*, void*);
struct FunctionSet : public IUserDataRoot {
  // flags for convert check
  static uint32_t tag_s_2_71828182846_;
  static uint32_t var_num_s_2_71828182846_;
  static string_view class_name_s_2_71828182846_;
  static IUserDataRoot::__FunctionTable__ function_table_s_2_71828182846_;

  // override meta functions
  const char* ClassName_2_71828182846() const override { return "FunctionSet"; }
  uint32_t tag_2_71828182846() const override { return tag_s_2_71828182846_; }
  uint32_t size_2_71828182846() const override { return var_num_s_2_71828182846_; }

  bool isinstance_2_71828182846(uint64_t tag) override {
    static std::initializer_list<uint64_t> all_tags = {FunctionSet::tag_s_2_71828182846_};
    return std::find(all_tags.begin(), all_tags.end(), tag) != all_tags.end();
  }

  std::initializer_list<string_view> VarNames_2_71828182846() const override {
    static std::initializer_list<string_view> __var_names_s__ = {};
    return __var_names_s__;
  }

  const ska::flat_hash_map<string_view, int64_t>& VarTable_2_71828182846() const override {
    static ska::flat_hash_map<string_view, int64_t> __var_table_s__ = {
    };
    return __var_table_s__;
  }

  // member vars

  // Object pointer
  Object* self_node_ptr_2_71828182846 = nullptr;

  // override GetVar_2_71828182846 functions
  RTView GetVar_2_71828182846(int64_t idx) const override {
    switch (idx) {
    default: { THROW_PY_IndexError("index overflow"); return nullptr; } break;

    }
  }
  // override SetVar_2_71828182846 functions
  void SetVar_2_71828182846(int64_t idx, const Any& val) override {
    switch (idx) {
    default: { THROW_PY_IndexError("index overflow"); } break;

    }
  }

  // virtual methods
  virtual RTValue __init__(void* handle_2_71828182846=((void*)(int64_t)0));
  virtual RTValue get_score(const DragonflyContext_SharedView& ctx);
};

// flags for convert check
uint32_t FunctionSet::tag_s_2_71828182846_ = 1477043230;
uint32_t FunctionSet::var_num_s_2_71828182846_ = 0;
string_view FunctionSet::class_name_s_2_71828182846_ = "FunctionSet";
IUserDataRoot::__FunctionTable__ FunctionSet::function_table_s_2_71828182846_ = IUserDataRoot::InitFuncTable_2_71828182846(&__matxscript_func_registry__FunctionSet, "FunctionSet");

struct FunctionSet_SharedView: public IUserDataSharedViewRoot {
  // member var
  FunctionSet *ptr;
  // constructor
  FunctionSet_SharedView(FunctionSet *ptr, UserDataRef ref) : ptr(ptr), IUserDataSharedViewRoot(std::move(ref)) {}
  FunctionSet_SharedView(FunctionSet *ptr) : ptr(ptr) {}
  FunctionSet_SharedView() : ptr(nullptr) {}
  FunctionSet_SharedView(const matxscript::runtime::Any& ref) : FunctionSet_SharedView(MATXSCRIPT_TYPE_AS_V2(ref, UserDataRef, "FunctionSet")) {}
  // UserDataRef
  FunctionSet_SharedView(UserDataRef ref) {
    IUserDataRoot* base_ud_ptr = static_cast<IUserDataRoot*>(ref.check_codegen_ptr("FunctionSet"));
    if(!base_ud_ptr->isinstance_2_71828182846(FunctionSet::tag_s_2_71828182846_)) {THROW_PY_TypeError("expect 'FunctionSet' but get '", base_ud_ptr->ClassName_2_71828182846(), "'");}
    ptr = static_cast<FunctionSet*>(base_ud_ptr);
    ud_ref = std::move(ref);
  }
  FunctionSet* operator->() const { return ptr; }
  template <typename T, typename = typename std::enable_if<std::is_convertible<UserDataRef, T>::value>::type>
  operator T() const {return ud_ref;}
};

void FunctionSet_F__deleter__(ILightUserData* ptr) { delete static_cast<FunctionSet*>(ptr); }
void* FunctionSet_F__placement_new__(void* buf) { return new (buf) FunctionSet; }
void FunctionSet_F__placement_del__(ILightUserData* ptr) { static_cast<FunctionSet*>(ptr)->FunctionSet::~FunctionSet(); }
FunctionSet_SharedView FunctionSet__F___init___wrapper(void* handle_2_71828182846) {
  static auto buffer_size = UserDataRef::GetInternalBufferSize();
  if (buffer_size < sizeof(FunctionSet)) {
    auto self = new FunctionSet;
    self->function_table_2_71828182846_ = &FunctionSet::function_table_s_2_71828182846_;
    FunctionSet__F___init__(self,  handle_2_71828182846);
    UserDataRef self_ref(FunctionSet::tag_s_2_71828182846_, FunctionSet::var_num_s_2_71828182846_, self, FunctionSet_F__deleter__, __matxscript_module_ctx);
    self->self_node_ptr_2_71828182846 = (Object*)(self_ref.get());
    return self_ref;
  } else {
    UserDataRef self(FunctionSet::tag_s_2_71828182846_, FunctionSet::var_num_s_2_71828182846_, sizeof(FunctionSet), FunctionSet_F__placement_new__, FunctionSet_F__placement_del__, __matxscript_module_ctx);
    FunctionSet_SharedView self_view((FunctionSet*)self.ud_ptr_nocheck());
    self_view->function_table_2_71828182846_ = &FunctionSet::function_table_s_2_71828182846_;
    FunctionSet__F___init__(self_view,  handle_2_71828182846);
    self_view->self_node_ptr_2_71828182846 = (Object*)(self.get());
    return self;
  }
}

RTValue FunctionSet::__init__(void* handle_2_71828182846) {  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
  this->session_handle_2_71828182846_ = handle_2_71828182846;  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
  return (None);
}

MATX_DLL RTValue FunctionSet__F___init__(const FunctionSet_SharedView& self, void* handle_2_71828182846) {  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
  return (self->__init__(handle_2_71828182846));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
}

int FunctionSet__F___init____c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[1] {"self"};
    KwargsUnpackHelper helper("__init__", arg_names, 1, nullptr, 0);
    RTView pos_args[1];
    helper.unpack(pos_args, args, num_args);  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
    auto ret = FunctionSet__F___init__(FunctionSet_SharedView(static_cast<const Any&>(pos_args[0])), resource_handle);  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 1: {
        auto ret = FunctionSet__F___init__(FunctionSet_SharedView(static_cast<const Any&>(args_t[0])), resource_handle);  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"FunctionSet.py\", line 5, in __init__\n", "__init__() takes 1 positional arguments but ", num_args, " were given");} break;  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
    }
  }

  return 0;
}

int FunctionSet__F___init___wrapper__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  switch(num_args) {
    case 0: {
      auto ret = FunctionSet__F___init___wrapper(resource_handle);  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
      (ret.operator RTValue()).MoveToCHost(out_ret_value);
    } break;
    default: {THROW_PY_TypeError("File \"FunctionSet.py\", line 5, in __init__\n", "__init__() takes 0 positional arguments but ", num_args, " were given");} break;  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:5
  }
  return 0;
}

RTValue FunctionSet::get_score(const DragonflyContext_SharedView& ctx) {  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
  ItemAttrGetter_SharedView video_quality_score_list_getter = (ctx->ItemAttrGetter(string_view("video_quality_score_list", 24)));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:10
  ItemAttrSetter_SharedView video_quality_score_setter = (ctx->ItemAttrSetter(string_view("video_quality_score", 19)));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:11
  ItemAttrSetter_SharedView video_cover_score_setter = (ctx->ItemAttrSetter(string_view("video_cover_score", 17)));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:12
  int64_t result_size = (ctx->GetItemNum());  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:14
  for (int64_t i_iter_ = (int64_t)0; i_iter_ < result_size; i_iter_ += (int64_t)1) {  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:15
    int64_t i = i_iter_;  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:15
    FTList<double> video_quality_score_list = (video_quality_score_list_getter->GetDoubleList(i));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:16
    (void)video_quality_score_setter->SetDouble(i, (video_quality_score_list).get_item(((int64_t)0)));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:17
    (void)video_cover_score_setter->SetDouble(i, (video_quality_score_list).get_item(((int64_t)1)));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:18
  }
  return (None);
}

MATX_DLL RTValue FunctionSet__F_get_score(const FunctionSet_SharedView& self, const DragonflyContext_SharedView& ctx) {  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
  return (self->get_score(ctx));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
}

int FunctionSet__F_get_score__c_api(MATXScriptAny* args, int num_args, MATXScriptAny* out_ret_value, void* resource_handle = nullptr)
{
  TArgs args_t(args, num_args);

  if (num_args > 0 && args[num_args - 1].code == TypeIndex::kRuntimeKwargs) {
    string_view arg_names[2] {"self", "ctx"};
    KwargsUnpackHelper helper("get_score", arg_names, 2, nullptr, 0);
    RTView pos_args[2];
    helper.unpack(pos_args, args, num_args);  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
    auto ret = FunctionSet__F_get_score(FunctionSet_SharedView(static_cast<const Any&>(pos_args[0])), DragonflyContext_SharedView(static_cast<const Any&>(pos_args[1])));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
    RTValue(std::move(ret)).MoveToCHost(out_ret_value);
  } else {
    switch(num_args) {
      case 2: {
        auto ret = FunctionSet__F_get_score(FunctionSet_SharedView(static_cast<const Any&>(args_t[0])), DragonflyContext_SharedView(static_cast<const Any&>(args_t[1])));  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
        RTValue(std::move(ret)).MoveToCHost(out_ret_value);
      } break;
      default: {THROW_PY_TypeError("File \"FunctionSet.py\", line 8, in get_score\n", "get_score() takes 2 positional arguments but ", num_args, " were given");} break;  // /media/disk1/fordata/web_server/project/py_compile_server/root_10.30.16.25_FunctionSet_clang/FunctionSet.py:8
    }
  }

  return 0;
}


} // namespace

extern "C" {

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__FunctionSet[] = {
    (MATXScriptBackendPackedCFunc)FunctionSet__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)FunctionSet__F___init____c_api,
    (MATXScriptBackendPackedCFunc)FunctionSet__F_get_score__c_api,
};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__FunctionSet = {
    "3\000FunctionSet__F___init___wrapper\000FunctionSet__F___init__\000FunctionSet__F_get_score\000",    __matxscript_func_array__FunctionSet,
};

} // extern C

extern "C" {

MATX_DLL MATXScriptBackendPackedCFunc __matxscript_func_array__[] = {
    (MATXScriptBackendPackedCFunc)FunctionSet__F___init___wrapper__c_api,
    (MATXScriptBackendPackedCFunc)FunctionSet__F_get_score__c_api,
};
MATX_DLL MATXScriptFuncRegistry __matxscript_func_registry__ = {
    "2\000FunctionSet__F___init___wrapper\000FunctionSet__F_get_score\000",    __matxscript_func_array__,
};

} // extern C

extern "C" {

MATX_DLL const char* __matxscript_closures_names__ = "2\000FunctionSet__F___init___wrapper\000FunctionSet__F___init__\000";

} // extern C

