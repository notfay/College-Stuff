package model;

public class Order {
	protected String username;
	protected String orderItem;
	
	public Order(String username, String orderItem) {
		super();
		this.username = username;
		this.orderItem = orderItem;
	}
	public String getUsername() {
		return username;
	}
	public void setUsername(String username) {
		this.username = username;
	}
	public String getOrderItem() {
		return orderItem;
	}
	public void setOrderItem(String orderItem) {
		this.orderItem = orderItem;
	}
	
}
