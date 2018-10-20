#pragma once

#include <array>
#include <memory>
#include <cstring>
#include <type_traits>
#include <typeinfo>
#include <typeindex>
#include <cassert>
#include <sstream>
#include <string>

namespace detail { namespace static_any {

struct move_tag {};
struct copy_tag {};

enum class operation_t { query_type, query_size, copy, move, destroy };

using function_ptr_t = void(*)(operation_t operation, void* this_ptr, void* other_ptr);

}}

template <std::size_t _N>
class static_any
{
public:
	template <typename _T>
	struct is_static_any : public std::false_type {};

	template <std::size_t _M>
	struct is_static_any<static_any<_M>> : public std::true_type {};

	template <class _T>
	static constexpr bool is_static_any_v = is_static_any<_T>::value;

	using size_type = std::size_t;

	static_any();
	~static_any();

	template <class _T,
			  class = std::enable_if_t<!is_static_any_v<std::decay_t<_T>>>>
	static_any(_T&&);

	static_any(const static_any&);

	template <std::size_t _M, class = std::enable_if_t<_M <= _N>>
	static_any(const static_any<_M>&);

	template <std::size_t _M, class = std::enable_if_t<_M <= _N>>
	static_any(static_any<_M>&&);

	template <class _T,
			  class = std::enable_if_t<!is_static_any_v<std::decay_t<_T>>>>
	static_any& operator=(_T&& t);

	static_any& operator=(const static_any& any)
	{
		assign_from_any(any);
		return *this;
	}

	template <std::size_t _M, class = std::enable_if_t<_M <= _N>>
	static_any& operator=(const static_any<_M>& any)
	{
		assign_from_any(any);
		return *this;
	}

	template <std::size_t _M, class = std::enable_if_t<_M <= _N>>
	static_any& operator=(static_any<_M>&& any)
	{
		assign_from_any(std::move(any));
		return *this;
	}

	void reset();

	template <class _T>
	const _T& get() const;

	template <class _T>
	_T& get();

	template <class _T>
	bool has() const;

	const std::type_info& type() const;

	bool empty() const;

	size_type size() const;

	static constexpr size_type capacity();

	template <class _T, class... Args>
	void emplace(Args&&... args);

private:
	using operation_t = detail::static_any::operation_t;
	using function_ptr_t = detail::static_any::function_ptr_t;

	template <class _T>
	void copy_or_move(_T&& t);

	template <class _T>
	void assign_from_any(_T&&);

	template <std::size_t _M, class CopyOrMoveTag>
	void assign_from_any(const static_any<_M>&, CopyOrMoveTag);

	const std::type_info& query_type() const;

	size_type query_size() const;

	void destroy();

	template <class _T>
	const _T* as() const;

	template <class _T>
	_T* as();

	template <class _RefT>
	void call_copy_or_move(void* this_void_ptr, void* other_void_ptr);

	void call_operation(const function_ptr_t& function, void* this_void_ptr, void* other_void_ptr, detail::static_any::move_tag);

	void call_operation(const function_ptr_t& function, void* this_void_ptr, void* other_void_ptr, detail::static_any::copy_tag);

	template <class _T>
	void copy_or_move_from_another(_T&&);

	std::array<char, _N> __buff;
	function_ptr_t __function{};

	template <std::size_t _S>
	friend class static_any;

	template <class _ValueT, std::size_t _S>
	friend _ValueT* any_cast(static_any<_S>*);

	template <class _ValueT, std::size_t _S>
	friend _ValueT& any_cast(static_any<_S>&);
};

namespace detail { namespace static_any {

template <class _T>
static void operation(operation_t operation, void* ptr1, void* ptr2)
{
	_T* this_ptr = reinterpret_cast<_T*>(ptr1);

	switch(operation)
	{
	case operation_t::query_type:
	{
		*reinterpret_cast<const std::type_info**>(ptr1) = &typeid(_T);
		break;
	}
	case operation_t::query_size:
	{
		*reinterpret_cast<std::size_t*>(ptr1) = sizeof(_T);
		break;
	}
	case operation_t::copy:
	{
		_T* other_ptr = reinterpret_cast<_T*>(ptr2);
		assert(this_ptr);
		assert(other_ptr);
		new(this_ptr)_T(*other_ptr);
		break;
	}
	case operation_t::move:
	{
		_T* other_ptr = reinterpret_cast<_T*>(ptr2);
		assert(this_ptr);
		assert(other_ptr);
		new(this_ptr)_T(std::move(*other_ptr));
		break;
	}
	case operation_t::destroy:
	{
		assert(this_ptr);
		this_ptr->~_T();
		break;
	}
	}
}

template <class _T>
static function_ptr_t get_function_for_type()
{
	return &static_any::operation<std::remove_cv_t<std::remove_reference_t<_T>>>;
}

}}

template <std::size_t _N>
static_any<_N>::static_any()
{}

template <std::size_t _N>
static_any<_N>::~static_any()
{
	destroy();
}

template <std::size_t _N>
template <class _T, class>
static_any<_N>::static_any(_T&& v)
{
	copy_or_move(std::forward<_T>(v));
}

template <std::size_t _N>
static_any<_N>::static_any(const static_any<_N>& another)
{
	copy_or_move_from_another(another);
}

template <std::size_t _N>
template <std::size_t _M, class>
static_any<_N>::static_any(const static_any<_M>& another)
{
	copy_or_move_from_another(another);
}

template <std::size_t _N>
template <std::size_t _M, class>
static_any<_N>::static_any(static_any<_M>&& another)
{
	copy_or_move_from_another(std::move(another));
}

template <std::size_t _N>
template <class _T, class>
static_any<_N>& static_any<_N>::operator=(_T&& t)
{
	static_assert(capacity() >= sizeof(_T), "_T is too big to be copied to static_any");

	using NonConstT = std::remove_cv_t<std::remove_reference_t<_T>>;
	NonConstT* non_const_t = const_cast<NonConstT*>(&t);

	static_any temp = std::move_if_noexcept(*this);

	try
	{
		destroy();
		assert(__function == nullptr);

		call_copy_or_move<_T&&>(__buff.data(), non_const_t);
	}
	catch(...)
	{
		*this = std::move(temp);
		throw;
	}

	__function = detail::static_any::get_function_for_type<_T>();
	return *this;
}

template <std::size_t _N>
void static_any<_N>::reset() { destroy(); }

template <std::size_t _N>
template <class _T>
bool static_any<_N>::has() const
{
	if (__function == detail::static_any::get_function_for_type<_T>())
	{
		return true;
	}
	else if (__function)
	{
		// need to try another, possibly more costly way, as we may compare types across DLL boundaries
		return std::type_index(typeid(_T)) == std::type_index(query_type());
	}
	return false;
}

template <std::size_t _N>
const std::type_info& static_any<_N>::type() const
{
	if (empty())
		return typeid(void);
	else
		return query_type();
}

template <std::size_t _N>
bool static_any<_N>::empty() const { return __function == nullptr; }

template <std::size_t _N>
typename static_any<_N>::size_type static_any<_N>::size() const
{
	if (empty())
		return 0;
	else
		return query_size();
}

template <std::size_t _N>
constexpr typename static_any<_N>::size_type static_any<_N>::capacity()
{
	return _N;
}

template <std::size_t _N>
template <class _T, class... Args>
void static_any<_N>::emplace(Args&&... args)
{
	destroy();
	new(__buff.data()) _T(std::forward<Args>(args)...);
	__function = detail::static_any::get_function_for_type<_T>();
}

template <std::size_t _N>
template <class _T>
void static_any<_N>::copy_or_move(_T&& t)
{
	static_assert(capacity() >= sizeof(_T), "_T is too big to be copied to static_any");
	assert(__function == nullptr);

	using NonConstT = std::remove_cv_t<std::remove_reference_t<_T>>;
	NonConstT* non_const_t = const_cast<NonConstT*>(&t);

	try {
		call_copy_or_move<_T&&>(__buff.data(), non_const_t);
	}
	catch(...) {
		throw;
	}

	__function = detail::static_any::get_function_for_type<_T>();
}

template <std::size_t _N>
template <class _T>
void static_any<_N>::assign_from_any(_T&& t)
{
	using CopyOrMoveTag = typename std::conditional<
		std::is_rvalue_reference<_T&&>::value,
			detail::static_any::move_tag,
			detail::static_any::copy_tag
		>::type;

	assign_from_any(std::forward<_T>(t), CopyOrMoveTag{});
}

template <std::size_t _N>
template <std::size_t _M, class CopyOrMoveTag>
void static_any<_N>::assign_from_any(const static_any<_M>& another, CopyOrMoveTag)
{
	if (another.__function == nullptr)
		return;

	static_any temp = std::move_if_noexcept(*this);
	void* other_data = reinterpret_cast<void*>(const_cast<char*>(another.__buff.data()));

	try {
		destroy();
		assert(__function == nullptr);

		call_operation(another.__function, __buff.data(), other_data, CopyOrMoveTag{});
	}
	catch(...) {
		*this = std::move(temp);
		throw;
	}

	__function= another.__function;
}

template <std::size_t _N>
const std::type_info& static_any<_N>::query_type() const
{
	assert(__function != nullptr);
	const std::type_info* ti ;
	__function(operation_t::query_type, &ti, nullptr);
	return *ti;
}

template <std::size_t _N>
typename static_any<_N>::size_type static_any<_N>::query_size() const
{
	assert(__function != nullptr);
	std::size_t size;
	__function(operation_t::query_size, &size, nullptr);
	return size;
}

template <std::size_t _N>
void static_any<_N>::destroy()
{
	if (__function)
	{
		void* not_used = nullptr;
		__function(operation_t::destroy, __buff.data(), not_used);
		__function = nullptr;
	}
}

template <std::size_t _N>
template <class _T>
const _T* static_any<_N>::as() const
{
	return reinterpret_cast<const _T*>(__buff.data());
}

template <std::size_t _N>
template <class _T>
_T* static_any<_N>::as()
{
	return reinterpret_cast<_T*>(__buff.data());
}

template <std::size_t _N>
template <class _RefT>
void static_any<_N>::call_copy_or_move(void* this_void_ptr, void* other_void_ptr)
{
	using Tag = typename std::conditional<std::is_rvalue_reference<_RefT&&>::value,
				detail::static_any::move_tag,
				detail::static_any::copy_tag>::type;

	auto function = detail::static_any::get_function_for_type<_RefT>();
	call_operation(function, this_void_ptr, other_void_ptr, Tag{});
}

template <std::size_t _N>
void static_any<_N>::call_operation(const function_ptr_t& function, void* this_void_ptr, void* other_void_ptr, detail::static_any::move_tag)
{
	function(operation_t::move, this_void_ptr, other_void_ptr);
}

template <std::size_t _N>
void static_any<_N>::call_operation(const function_ptr_t& function, void* this_void_ptr, void* other_void_ptr, detail::static_any::copy_tag)
{
	function(operation_t::copy, this_void_ptr, other_void_ptr);
}

template <std::size_t _N>
template <class _T>
void static_any<_N>::copy_or_move_from_another(_T&& another)
{
	assert(__function == nullptr);

	if (another.__function == nullptr)
	{
		return;
	}

	using Tag = typename std::conditional<std::is_rvalue_reference<_T&&>::value,
				detail::static_any::move_tag,
				detail::static_any::copy_tag>::type;

	void* other_data = reinterpret_cast<void*>(const_cast<char*>(another.__buff.data()));

	try {
		call_operation(another.__function, __buff.data(), other_data, Tag{});
	}
	catch(...) {
		throw;
	}

	__function= another.__function;
}

class bad_any_cast : public std::bad_cast
{
public:
	explicit bad_any_cast(const std::type_info& from,
						  const std::type_info& to);
	virtual ~bad_any_cast();

	const std::type_info& stored_type() const { return __from; }
	const std::type_info& target_type() const { return __to; }

	const char* what() const noexcept override
	{
		return __reason.c_str();
	}

private:
	const std::type_info& __from;
	const std::type_info& __to;
	std::string __reason;
};

bad_any_cast::bad_any_cast(const std::type_info& from,
						   const std::type_info& to) :
	__from(from),
	__to(to)
{
	std::ostringstream oss;
	oss << "failed conversion using any_cast: stored type "
		<< from.name()
		<< ", trying to cast to "
		<< to.name();
	__reason = oss.str();
}

bad_any_cast::~bad_any_cast() {}

template <class _ValueT,
		  std::size_t _S>
inline _ValueT* any_cast(static_any<_S>* a)
{
	if (!a->template has<_ValueT>())
		return nullptr;

	return a->template as<_ValueT>();
}

template <class _ValueT,
		  std::size_t _S>
inline const _ValueT* any_cast(const static_any<_S>* a)
{
	return any_cast<const _ValueT>(const_cast<static_any<_S>*>(a));
}

template <class _ValueT,
		  std::size_t _S>
inline _ValueT& any_cast(static_any<_S>& a)
{
	if (!a.template has<_ValueT>())
		throw bad_any_cast(a.type(), typeid(_ValueT));

	return *a.template as<_ValueT>();
}

template <class _ValueT,
		  std::size_t _S>
inline const _ValueT& any_cast(const static_any<_S>& a)
{
	return any_cast<const _ValueT>(const_cast<static_any<_S>&>(a));
}

template <std::size_t _S>
template <class _T>
const _T& static_any<_S>::get() const
{
	return any_cast<_T>(*this);
}

template <std::size_t _S>
template <class _T>
_T& static_any<_S>::get()
{
	return any_cast<_T>(*this);
}


template <std::size_t _N>
class static_any_t
{
public:
	using size_type = std::size_t;

	static constexpr size_type capacity() { return _N; }

	static_any_t() = default;
	static_any_t(const static_any_t&) = default;

	template <class _ValueT>
	static_any_t(_ValueT&& t)
	{
		copy(std::forward<_ValueT>(t));
	}

	template <class _ValueT>
	static_any_t& operator=(_ValueT&& t)
	{
		copy(std::forward<_ValueT>(t));
		return *this;
	}

	template <class _ValueT>
	_ValueT& get() { return *reinterpret_cast<_ValueT*>(__buff.data()); }

	template <class _ValueT>
	const _ValueT& get() const { return *reinterpret_cast<const _ValueT*>(__buff.data()); }

private:
	template <class _ValueT>
	void copy(_ValueT&& t)
	{
		using NonConstT = std::remove_cv_t<std::remove_reference_t<_ValueT>>;

#if __GNUG__ && __GNUC__ < 5
		static_assert(std::has_trivial_copy_constructor<NonConstT>::value, "_ValueT is not trivially copyable");
#else
		static_assert(std::is_trivially_copyable<NonConstT>::value, "_ValueT is not trivially copyable");
#endif

		static_assert(capacity() >= sizeof(_ValueT), "_ValueT is too big to be copied to static_any");

		std::memcpy(__buff.data(), reinterpret_cast<char*>(&t), sizeof(_ValueT));
	}

	std::array<char, _N> __buff;
};
